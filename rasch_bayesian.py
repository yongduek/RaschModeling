import pandas as pd
import numpy as np
import cmdstanpy
import os
import sys

# Ensure compiled model can be found or recompiled
# cmdstanpy automatically manages compilation if stan file changes

def run_bayesian_estimation():
    # 1. Import Data
    try:
        from readBLOT import read_blot_data
    except ImportError:
        sys.path.append(os.getcwd())
        from readBLOT import read_blot_data
        
    print("Loading Data...")
    df = read_blot_data('blot.txt')
    
    # 2. Reshape Data for Stan (Long Format)
    item_cols = [c for c in df.columns if c != 'PersonID']
    # Check if we have valid items
    if len(item_cols) == 0:
        print("No item columns found.")
        return

    # Handle duplicate item names by appending index to make them unique
    # This also ensures melt works correctly
    original_item_names = list(item_cols)
    unique_item_names = [f"{name}_{i+1}" for i, name in enumerate(original_item_names)]
    
    # Create a mapping from unique name to original name for reporting
    name_map = dict(zip(unique_item_names, original_item_names))
    
    # Rename columns in df
    df_renamed = df.copy()
    # We only rename item cols. 
    # Current columns: PersonID, Item1, Item2...
    # We need to construct full column list
    new_columns = ['PersonID'] + unique_item_names
    # Check if columns align. df might have duplicate names, so direct rename map is tricky.
    # Set columns directly
    if len(df_renamed.columns) == len(new_columns):
        df_renamed.columns = new_columns
    else:
        # Fallback if structure is unexpected
        print("Error: DataFrame structure mismatch.")
        return

    # Persons 1..N
    persons = df_renamed['PersonID'].unique()
    person_map = {pid: i+1 for i, pid in enumerate(persons)}
    n_persons = len(persons)
    
    # Items 1..K
    items = unique_item_names
    item_map = {item: i+1 for i, item in enumerate(items)}
    n_items = len(items)
    
    print(f"Dataset: {n_persons} persons, {n_items} items")
    
    # Melt dataframe to long format
    # Use the NEW unique item names
    df_long = df_renamed.melt(id_vars='PersonID', value_vars=unique_item_names, var_name='Item', value_name='Response')
    
    # Drop missing values
    df_long['Response'] = pd.to_numeric(df_long['Response'], errors='coerce')
    df_long = df_long.dropna(subset=['Response'])
    df_long['Response'] = df_long['Response'].astype(int)
    
    # Map IDs to integers
    df_long['jj'] = df_long['PersonID'].map(person_map)
    df_long['kk'] = df_long['Item'].map(item_map)
    
    # Prepare Data Dictionary
    stan_data = {
        'N_obs': len(df_long),
        'N_persons': n_persons,
        'N_items': n_items,
        'jj': df_long['jj'].values,
        'kk': df_long['kk'].values,
        'y': df_long['Response'].values
    }
    
    # 3. Initialize and Compile Model
    stan_file = 'rasch.stan'
    print(f"\nCompiling {stan_file}...")
    try:
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    except Exception as e:
        print(f"Error initializing CmdStanModel: {e}")
        return

    # 4. Sample
    print("\nStarting MCMC Sampling...")
    # Increase sampling slightly for better PPC resolution
    fit = model.sample(data=stan_data, 
                       chains=4, 
                       iter_warmup=500, 
                       iter_sampling=500, 
                       show_progress=True)
    
    print("\nSampling completed.")
    
    # --- PPC Analysis ---
    print("\nPerforming Posterior Predictive Checks (PPC)...")
    
    # Extract y_rep: Shape (Draws, N_obs)
    # This might be large, so handle carefully
    y_rep = fit.stan_variable('y_rep')
    # shape: (2000, 5250) roughly
    
    # 1. Item Raw Score Compatibility
    # Calculate observed item raw scores
    obs_item_scores = df_long.groupby('kk')['Response'].sum().sort_index().values
    
    # Calculate replicated item raw scores for each draw
    # We need to map linear index back to item/person
    # To do this efficiently in numpy:
    # Create an index mapper matrix matching y_rep columns
    item_indices = df_long['kk'].values - 1 # 0-based
    
    # rep_item_scores shape: (Draws, N_items)
    n_draws = y_rep.shape[0]
    rep_item_scores = np.zeros((n_draws, n_items))
    
    # Vectorized sum by group is tricky in pure numpy without pandas. 
    # But looping over items is fast enough (35 items)
    for k in range(n_items):
        # find columns in y_rep corresponding to item k+1
        cols = (item_indices == k)
        # Sum across those columns
        rep_item_scores[:, k] = np.sum(y_rep[:, cols], axis=1)
        
    # Calculate PPP-values (Posterior Predictive P-values)
    # Proportion of Rep Scores > Obs Score
    # Two-sided p-value: 2 * min(P(Rep > Obs), P(Rep < Obs))
    ppp_values = []
    print("\nPosterior Predictive Check - Item Raw Scores:")
    print(f"{'Item':<15} {'ObsScore':<10} {'RepMean':<10} {'PPP-Val':<10}")
    
    ppc_data = []
    
    for k in range(n_items):
        obs = obs_item_scores[k]
        reps = rep_item_scores[:, k]
        rep_mean = np.mean(reps)
        
        # P(Rep >= Obs)
        p_greater = np.mean(reps >= obs)
        # P(Rep <= Obs)
        p_less = np.mean(reps <= obs)
        
        # Two-sided p
        ppp = 2 * min(p_greater, p_less)
        
        item_name = original_item_names[k] # Assumes ordering matches 1..K
        print(f"{item_name:<15} {obs:<10} {rep_mean:<10.1f} {ppp:<10.3f}")
        
        ppc_data.append({
            'ItemName': item_name,
            'ObservedScore': obs,
            'ExpectedScore_PPC': rep_mean,
            'PPP_Value': ppp
        })
        
    ppc_df = pd.DataFrame(ppc_data)
    ppc_df.to_csv('rasch_ppc_item_fit.csv', index=False)
    print("Saved PPC results to rasch_ppc_item_fit.csv")
    
    # 5. Extract Results
    summary = fit.summary()
    
    # Extract Beta (Item Difficulties)
    # Rows starting with 'beta['
    beta_rows = [idx for idx in summary.index if idx.startswith('beta[')]
    item_results = summary.loc[beta_rows, ['Mean', 'StdDev', '5%', '95%', 'R_hat']]
    
    # Add Item Names
    params_idx = []
    # summary index is beta[1], beta[2]... 
    # Use the original item names list, which matches 1..K order
    item_results['ItemName'] = original_item_names
    item_results['UniqueName'] = unique_item_names
    item_results['ItemIndex'] = range(1, n_items + 1)
    
    # Rename columns for clarity
    item_results = item_results.rename(columns={'Mean': 'Difficulty_Bayes', 'StdDev': 'SE_Bayes'})
    
    # --- Compute Fit Statistics ---
    print("Computing Fit Statistics...")
    
    # 1. Prepare Vectors
    # Beta Vector (1..K)
    beta_vec = np.zeros(n_items)
    for i in range(n_items):
        key = f'beta[{i+1}]'
        beta_vec[i] = item_results.loc[key, 'Difficulty_Bayes']
        
    # Theta Vector (1..N)
    theta_vec = np.zeros(n_persons)
    for i in range(n_persons):
        key = f'theta[{i+1}]'
        # person_results used 'Mean' originally, make sure we use the renamed column if we renamed it
        # But person_results is processed LATER in the script.
        # We need to access summary directly or extract it here.
        # Let's extract theta means from summary now.
        theta_vec[i] = summary.loc[key, 'Mean']

    # 2. Data Matrix X
    # Reconstruct matrix respecting the person/item order
    df_indexed = df_renamed.set_index('PersonID')
    # p_ids matches 1..N
    sorted_persons = sorted(person_map.items(), key=lambda x: x[1])
    p_ids = [x[0] for x in sorted_persons]
    
    # Select columns matching 1..K (unique_item_names)
    X = df_indexed.loc[p_ids, unique_item_names].apply(pd.to_numeric, errors='coerce').values
    mask = ~np.isnan(X)
    
    # 3. Probability Matrix & Residuals
    # logit[n,i] = theta[n] - beta[i]
    logit_matrix = theta_vec[:, np.newaxis] - beta_vec[np.newaxis, :]
    logit_matrix = np.clip(logit_matrix, -30, 30) # Avoid overflow
    
    P_matrix = np.exp(logit_matrix) / (1.0 + np.exp(logit_matrix))
    W_matrix = P_matrix * (1.0 - P_matrix)
    
    Residuals = X - P_matrix
    
    # Zero out masked
    Residuals[~mask] = 0
    W_matrix[~mask] = 0
    
    # Std Residuals
    # Avoid div/0
    W_safe = W_matrix.copy()
    W_safe[W_safe < 1e-10] = 1.0 # Dummy
    StdRes = Residuals / np.sqrt(W_safe)
    StdRes[~mask] = 0

    # 4. Infit / Outfit MNSQ
    N_valid = np.sum(mask, axis=0) # (K,)
    sum_W = np.sum(W_matrix, axis=0)
    
    infit_msq = np.sum(Residuals**2, axis=0) / (sum_W + 1e-12)
    outfit_msq = np.sum(StdRes**2, axis=0) / N_valid
    
    # 5. t-statistics
    # Outfit Var
    term_outfit = (1.0 / W_safe) - 4.0
    term_outfit[~mask] = 0
    outfit_var = np.sum(term_outfit, axis=0) / (N_valid**2)
    outfit_var = np.maximum(outfit_var, 1e-10)
    
    outfit_t = (np.power(outfit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(outfit_var)) + (np.sqrt(outfit_var) / 3.0)
    
    # Infit Var
    term_infit = W_matrix - 4.0 * W_matrix**2
    term_infit[~mask] = 0
    infit_var = np.sum(term_infit, axis=0) / (sum_W**2)
    infit_var = np.maximum(infit_var, 1e-10)
    
    infit_t = (np.power(infit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(infit_var)) + (np.sqrt(infit_var) / 3.0)
    
    # 6. Point Measure Corr
    pm_corrs = []
    for i in range(n_items):
        valid_idx = mask[:, i]
        if np.sum(valid_idx) > 1:
            r = np.corrcoef(X[valid_idx, i], theta_vec[valid_idx])[0, 1]
        else:
            r = np.nan
        pm_corrs.append(r)
        
    # Add to DataFrame
    keys = [f'beta[{i+1}]' for i in range(n_items)]
    item_results.loc[keys, 'InfitMNSQ'] = infit_msq
    item_results.loc[keys, 'OutfitMNSQ'] = outfit_msq
    item_results.loc[keys, 'Infit_t'] = infit_t
    item_results.loc[keys, 'Outfit_t'] = outfit_t
    item_results.loc[keys, 'PtMeasureCorr'] = pm_corrs

    # Sort by Difficulty
    item_results = item_results.sort_values('Difficulty_Bayes', ascending=False)
    
    # Save Item Results
    cols_to_save = ['ItemIndex', 'ItemName', 'Difficulty_Bayes', 'SE_Bayes', 
                    'InfitMNSQ', 'OutfitMNSQ', 'Infit_t', 'Outfit_t', 'PtMeasureCorr',
                    '5%', '95%', 'R_hat']
    item_results[cols_to_save].to_csv('rasch_bayes_items.csv', index=False)
    
    # Extract Theta (Person Measures)
    theta_rows = [idx for idx in summary.index if idx.startswith('theta[')]
    person_results = summary.loc[theta_rows, ['Mean', 'StdDev']]
    
    # Map back to Person IDs
    # Assuming theta[1] corresponds to person_map[ID]=1
    sorted_persons = sorted(person_map.items(), key=lambda x: x[1])
    p_ids = [x[0] for x in sorted_persons]
    
    person_results['PersonID'] = p_ids
    person_results = person_results.rename(columns={'Mean': 'Measure_Bayes', 'StdDev': 'SE_Bayes'})
    
    # Save Person Results
    person_results[['PersonID', 'Measure_Bayes', 'SE_Bayes']].to_csv('rasch_bayes_persons.csv', index=False)

    print("\nBayesian Estimation Results (Head):")
    print(item_results[['ItemName', 'Difficulty_Bayes', 'SE_Bayes']].head())
    print(f"\nSaved results to rasch_bayes_items.csv and rasch_bayes_persons.csv")

if __name__ == "__main__":
    run_bayesian_estimation()
