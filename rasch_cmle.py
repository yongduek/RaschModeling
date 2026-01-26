import pandas as pd
import numpy as np
from scipy.optimize import root, newton
import os
import sys

# Import read_blot_data from existing file
try:
    from readBLOT import read_blot_data
except ImportError:
    # Ensure current directory is in path
    sys.path.append(os.getcwd())
    try:
        from readBLOT import read_blot_data
    except ImportError:
        print("Error: Could not import read_blot_data from readBLOT.py")
        sys.exit(1)

def get_symmetric_functions(betas):
    """
    Computes elementary symmetric functions gamma for Rasch model.
    betas: array of item parameters (epsilon = exp(-delta))
    Returns gamma array where gamma[r] is the value for raw score r.
    """
    n = len(betas)
    gamma = np.zeros(n + 1)
    gamma[0] = 1.0
    
    # Recursive calculation: gamma_r(new) = gamma_r(old) + beta * gamma_{r-1}(old)
    for b in betas:
        # Vectorized update. correspond to:
        # for r in range(n, 0, -1): gamma[r] += b * gamma[r-1]
        # We must use values from previous step, so numpy slice works as it creates a temp copy for RHS
        gamma[1:] += b * gamma[:-1]
        
    return gamma

def get_expected_scores(betas, score_counts):
    """
    Computes expected item scores S_i = sum_r n_r * P(x_vi=1|r)
    using CML formulas.
    """
    n_items = len(betas)
    gammas = get_symmetric_functions(betas)
    
    exp_scores = np.zeros(n_items)
    
    # Need to verify no zero division
    # gammas[r] cannot be zero if betas > 0
    
    for i in range(n_items):
        beta = betas[i]
        
        # Calculate gamma without item i (gamma_without_i)
        # gamma_r = gamma_without_i[r] + beta * gamma_without_i[r-1]
        # => gamma_without_i[r] = gamma_r - beta * gamma_without_i[r-1]
        
        g_no_i = np.zeros(n_items + 1)
        g_no_i[0] = 1.0
        
        # Recover partial gammas
        for r in range(1, n_items + 1):
            g_no_i[r] = gammas[r] - beta * g_no_i[r-1]
            
        expected_val = 0.0
        for r, count in score_counts.items():
            if count > 0 and 0 < r < n_items: # Only valid scores contribute to CML
                # P(x_vi=1 | r) = (beta_i * gamma_{r-1}^(without i)) / gamma_r
                term = (beta * g_no_i[r-1]) / gammas[r]
                expected_val += count * term
        
        exp_scores[i] = expected_val
        
    return exp_scores

def solve_theta(score, item_difficulties):
    """
    Estimates theta for a given raw score using MLE (Newton-Raphson).
    """
    # Boundary cases
    if score <= 0:
        # Extreme score handler: usually requires bias correction or Bayesian prior
        # Here we return a heuristic low value relative to easiest item
        return np.min(item_difficulties) - 2.5 # approx - infinity
    if score >= len(item_difficulties):
        return np.max(item_difficulties) + 2.5 # approx + infinity
        
    def func(theta):
        # sum of probabilities - score
        # prob = exp(theta - delta) / (1 + exp(theta - delta))
        d = theta - item_difficulties
        probs = np.exp(d) / (1 + np.exp(d))
        return np.sum(probs) - score
    
    def fprime(theta):
        # Derivative of sum of probs with respect to theta
        # d/dtheta P = P * (1-P)
        d = theta - item_difficulties
        probs = np.exp(d) / (1 + np.exp(d))
        return np.sum(probs * (1 - probs))

    try:
        # Newton-Raphson
        theta_est = newton(func, x0=0.0, fprime=fprime, maxiter=50)
        return theta_est
    except RuntimeError:
        return np.nan

def rasch_cmle(datafile='blot.txt'):
    print(f"Loading Data from {datafile}...")
    df = read_blot_data(datafile)
    
    if df is None or df.empty:
        print("Error: Empty or missing dataframe.")
        return None, None

    # Extract Item Matrix
    # Avoid duplicate column selection issues by using iloc
    # PersonID is column 0
    item_cols = df.columns[1:] # Keep original names (even if duplicates)
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    print(f"Data Loaded. Shape: {X.shape}")
    
    # 1. Prepare Sufficient Statistics for CML
    person_scores = np.sum(X, axis=1)
    n_items = X.shape[1]
    n_persons = X.shape[0]
    
    # Filter for Calibration (remove perfect and zero scores)
    valid_mask = (person_scores > 0) & (person_scores < n_items)
    X_calib = X[valid_mask]
    
    if len(X_calib) == 0:
        print("No valid cases for CML estimation (all scores are 0 or max).")
        return None, None
        
    print(f"Calibration Sample: {len(X_calib)} persons (excluding extreme scores)")
    
    item_totals = np.sum(X_calib, axis=0)
    
    # Counts of each raw score in calibration sample
    # We need n_r for r in 1..k-1
    calib_scores = np.sum(X_calib, axis=1)
    # Using integer casting to ensure dictionary keys match indices
    calib_scores = calib_scores.astype(int)
    
    score_counts = {}
    for s in calib_scores:
        score_counts[s] = score_counts.get(s, 0) + 1
        
    # 2. CML Estimation of Item Parameters
    print("Estimating Item Parameters (CMLE)...")
    
    # Initial values: Logits of p-values
    p_values = item_totals / len(X_calib)
    p_values = np.clip(p_values, 0.001, 0.999) # Avoid 0/1
    logit_diff = np.log((1 - p_values) / p_values)
    
    # Center to mean 0
    logit_diff = logit_diff - np.mean(logit_diff)
    
    # Define objective function for root finder
    # We fix sum(delta) = 0.
    # Solve for N-1 deltas.
    
    def equations(delta_subset):
        # Reconstruct full delta vector
        last_delta = -np.sum(delta_subset)
        full_deltas = np.append(delta_subset, last_delta)
        
        # betas = exp(-delta)
        # For stability, work with exp(-delta)
        betas = np.exp(-full_deltas)
        
        exp_score_vec = get_expected_scores(betas, score_counts)
        
        # Difference between Observed Item Totals and Expected
        # We drop the last equation (redundant)
        residuals = (item_totals - exp_score_vec)
        return residuals[:-1]

    # Run Solver
    # Levenberg-Marquardt is robust
    sol = root(equations, logit_diff[:-1], method='lm', options={'ftol': 1e-6})
    
    if not sol.success:
        print("Warning: CMLE did not converge.")
        print(sol.message)
        
    final_deltas_subset = sol.x
    final_last_delta = -np.sum(final_deltas_subset)
    item_difficulties = np.append(final_deltas_subset, final_last_delta)
    
    # 3. Person Measurement (MLE)
    print("\nEstimating Person Measures...")
    
    # Map raw score to measure
    possible_scores = np.arange(n_items + 1)
    score_map = {}
    for s in possible_scores:
        t = solve_theta(s, item_difficulties)
        score_map[s] = t
        
    # Apply to full dataset
    full_scores = np.sum(X, axis=1).astype(int)
    person_measures = np.array([score_map[s] for s in full_scores])
    
    # 4. Compute Fit Statistics and Standard Errors
    print("Computing Fit Statistics...")
    
    # Calculate Probability Matrix P[n, i]
    # P = exp(theta_n - delta_i) / (1 + exp)
    # n_persons x n_items
    
    theta_matrix = person_measures[:, np.newaxis] # Column vector
    delta_matrix = item_difficulties[np.newaxis, :] # Row vector
    logit_matrix = theta_matrix - delta_matrix
    
    # Handle overflow/underflow in exp
    # Limit logits to reasonable range e.g. [-30, 30] to prev overflow
    logit_matrix = np.clip(logit_matrix, -30, 30)
    
    P_matrix = np.exp(logit_matrix) / (1.0 + np.exp(logit_matrix))
    W_matrix = P_matrix * (1.0 - P_matrix)
    
    # Item SE Approximation: 1 / sqrt(sum(W_ni)) over all persons
    # Note: Traditional SE uses the sample, effectively treating person params as fixed.
    item_se = 1.0 / np.sqrt(np.sum(W_matrix, axis=0))
    
    # Residuals
    # X is the data matrix (0/1)
    # Exclude extreme persons/missing data if any? X is clean 0/1 here.
    residuals = X - P_matrix
    std_residuals = residuals / np.sqrt(W_matrix) # Z_ni
    
    # Infit MNSQ = Sum(Res^2) / Sum(W)
    infit_msq = np.sum(residuals**2, axis=0) / np.sum(W_matrix, axis=0)
    
    # Outfit MNSQ = Sum(Z^2) / N
    # Use N as count of valid responses (here all N)
    outfit_msq = np.sum(std_residuals**2, axis=0) / n_persons
    
    # Point-Measure Correlation
    # Pearson corr between X[:, i] and person_measures
    pm_corrs = []
    for i in range(n_items):
        r = np.corrcoef(X[:, i], person_measures)[0, 1]
        pm_corrs.append(r)
    pm_corrs = np.array(pm_corrs)

    # Calculation of t-statistics (Wilson-Hilferty transformation)
    # Variance of the squared standardized residuals is needed.
    # q_infit formula approx (Wright & Masters/Linacre)
    # Var(InfitMNSQ) approx Q / Sum(W)^2 where Q = sum(W(1-4W)) ? No.
    # Standard approximation often used in Winsteps:
    # qi = sum(W_ni * (1 - W_ni) * (1 - 4W_ni)) # Wait, that's not quite right.
    # Variance of observation kurtosis term involved.
    
    # Simplified approach for t-stats (approximate):
    # For Outfit: Var(u_i) ~ sum(1/W_ni * k_ni) / N^2 ?? 
    # Use standard formulas:
    
    # For Outfit t:
    # q_i^2 = sum(C_ni / W_ni^2) where C_ni = kurtosis factor = W(1-4W) ??
    # Actually Var(z^2) = (1/W - 4). 
    # Var(Outfit) = sum((1/W_ni - 4)) / N^2
    
    outfit_var = np.sum((1.0 / W_matrix) - 4.0, axis=0) / (n_persons**2)
    # Clip to avoid negative var in extreme edge cases (unlikely with items)
    outfit_var = np.maximum(outfit_var, 1e-10)
    outfit_t = (np.power(outfit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(outfit_var)) + (np.sqrt(outfit_var) / 3.0)
    
    # For Infit t:
    # Var(Infit) = sum( (W - 4W^2) ) / (sum(W))^2
    infit_var = np.sum(W_matrix - 4.0 * W_matrix**2, axis=0) / (np.sum(W_matrix, axis=0)**2)
    infit_var = np.maximum(infit_var, 1e-10)
    infit_t = (np.power(infit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(infit_var)) + (np.sqrt(infit_var) / 3.0)
    
    # Combine results
    results = pd.DataFrame({
        'ItemIndex': np.arange(1, n_items + 1),
        'ItemName': item_cols,
        'Difficulty': item_difficulties,
        'SE': item_se,
        'InfitMNSQ': infit_msq,
        'OutfitMNSQ': outfit_msq,
        'Infit_t': infit_t,
        'Outfit_t': outfit_t,
        'PtMeasureCorr': pm_corrs
    })
    

    
    # Sort by Difficulty descending (largest first)
    results = results.sort_values(by='Difficulty', ascending=False)
    
    print("\nCalibration Results (First 5 items):")
    print(results.head())
    
    df_out = df[['PersonID']].copy()
    df_out['RawScore'] = full_scores
    df_out['Measure'] = person_measures
    
    return results, df_out

def rasch_cmle(datafile='blot.txt'):
    print(f"Loading Data from {datafile}...")
    df = read_blot_data(datafile)
    
    if df is None or df.empty:
        print("Error: Empty or missing dataframe.")
        return None, None
        
    # Extract Item Matrix
    # Avoid duplicate column selection issues by using iloc
    # PersonID is column 0
    item_cols = df.columns[1:] # Keep original names (even if duplicates)
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    print(f"Data Loaded. Shape: {X.shape}")
    
    results, df_out = estimate_rasch_cmle_from_matrix(X, item_cols, df[['PersonID']])
    
    # Save (optional, kept from original logic if needed, but commented out in return)
    return results, df_out

def estimate_rasch_cmle_from_matrix(X, item_cols=None, person_id_df=None):
    """
    Refactored function to estimate Rasch parameters from binary matrix X.
    """
    if item_cols is None:
        item_cols = [f'Item_{i+1}' for i in range(X.shape[1])]
    
    # 1. Prepare Sufficient Statistics for CML
    person_scores = np.sum(X, axis=1)
    n_items = X.shape[1]
    n_persons = X.shape[0]
    
    # Filter for Calibration (remove perfect and zero scores)
    valid_mask = (person_scores > 0) & (person_scores < n_items)
    X_calib = X[valid_mask]
    
    if len(X_calib) == 0:
        print("No valid cases for CML estimation (all scores are 0 or max).")
        return None, None
        
    print(f"Calibration Sample: {len(X_calib)} persons (excluding extreme scores)")
    
    item_totals = np.sum(X_calib, axis=0)
    
    # Counts of each raw score in calibration sample
    # We need n_r for r in 1..k-1
    calib_scores = np.sum(X_calib, axis=1)
    # Using integer casting to ensure dictionary keys match indices
    calib_scores = calib_scores.astype(int)
    
    score_counts = {}
    for s in calib_scores:
        score_counts[s] = score_counts.get(s, 0) + 1
        
    # 2. CML Estimation of Item Parameters
    print("Estimating Item Parameters (CMLE)...")
    
    # Initial values: Logits of p-values
    p_values = item_totals / len(X_calib)
    p_values = np.clip(p_values, 0.001, 0.999) # Avoid 0/1
    logit_diff = np.log((1 - p_values) / p_values)
    
    # Center to mean 0
    logit_diff = logit_diff - np.mean(logit_diff)
    
    # Define objective function for root finder
    # We fix sum(delta) = 0.
    # Solve for N-1 deltas.
    
    def equations(delta_subset):
        # Reconstruct full delta vector
        last_delta = -np.sum(delta_subset)
        full_deltas = np.append(delta_subset, last_delta)
        
        # betas = exp(-delta)
        # For stability, work with exp(-delta)
        betas = np.exp(-full_deltas)
        
        exp_score_vec = get_expected_scores(betas, score_counts)
        
        # Difference between Observed Item Totals and Expected
        # We drop the last equation (redundant)
        residuals = (item_totals - exp_score_vec)
        return residuals[:-1]

    # Run Solver
    # Levenberg-Marquardt is robust
    sol = root(equations, logit_diff[:-1], method='lm', options={'ftol': 1e-6})
    
    if not sol.success:
        print("Warning: CMLE did not converge.")
        print(sol.message)
        
    final_deltas_subset = sol.x
    final_last_delta = -np.sum(final_deltas_subset)
    item_difficulties = np.append(final_deltas_subset, final_last_delta)
    
    # 3. Person Measurement (MLE)
    print("\nEstimating Person Measures...")
    
    # Map raw score to measure
    possible_scores = np.arange(n_items + 1)
    score_map = {}
    for s in possible_scores:
        t = solve_theta(s, item_difficulties)
        score_map[s] = t
        
    # Apply to full dataset
    full_scores = np.sum(X, axis=1).astype(int)
    person_measures = np.array([score_map[s] for s in full_scores])
    
    # 4. Compute Fit Statistics and Standard Errors
    print("Computing Fit Statistics...")
    
    # Calculate Probability Matrix P[n, i]
    # P = exp(theta_n - delta_i) / (1 + exp)
    # n_persons x n_items
    
    theta_matrix = person_measures[:, np.newaxis] # Column vector
    delta_matrix = item_difficulties[np.newaxis, :] # Row vector
    logit_matrix = theta_matrix - delta_matrix
    
    # Handle overflow/underflow in exp
    # Limit logits to reasonable range e.g. [-30, 30] to prev overflow
    logit_matrix = np.clip(logit_matrix, -30, 30)
    
    P_matrix = np.exp(logit_matrix) / (1.0 + np.exp(logit_matrix))
    W_matrix = P_matrix * (1.0 - P_matrix)
    
    # Item SE Approximation: 1 / sqrt(sum(W_ni)) over all persons
    # Note: Traditional SE uses the sample, effectively treating person params as fixed.
    item_se = 1.0 / np.sqrt(np.sum(W_matrix, axis=0))
    
    # Residuals
    # X is the data matrix (0/1)
    # Exclude extreme persons/missing data if any? X is clean 0/1 here.
    residuals = X - P_matrix
    std_residuals = residuals / np.sqrt(W_matrix) # Z_ni
    
    # Infit MNSQ = Sum(Res^2) / Sum(W)
    infit_msq = np.sum(residuals**2, axis=0) / np.sum(W_matrix, axis=0)
    
    # Outfit MNSQ = Sum(Z^2) / N
    # Use N as count of valid responses (here all N)
    outfit_msq = np.sum(std_residuals**2, axis=0) / n_persons
    
    # Point-Measure Correlation
    # Pearson corr between X[:, i] and person_measures
    pm_corrs = []
    for i in range(n_items):
        r = np.corrcoef(X[:, i], person_measures)[0, 1]
        pm_corrs.append(r)
    pm_corrs = np.array(pm_corrs)

    # Calculation of t-statistics (Wilson-Hilferty transformation)
    # Variance of the squared standardized residuals is needed.
    # q_infit formula approx (Wright & Masters/Linacre)
    # Var(InfitMNSQ) approx Q / Sum(W)^2 where Q = sum(W(1-4W)) ? No.
    # Standard approximation often used in Winsteps:
    # qi = sum(W_ni * (1 - W_ni) * (1 - 4W_ni)) # Wait, that's not quite right.
    # Variance of observation kurtosis term involved.
    
    # Simplified approach for t-stats (approximate):
    # For Outfit: Var(u_i) ~ sum(1/W_ni * k_ni) / N^2 ?? 
    # Use standard formulas:
    
    # For Outfit t:
    # q_i^2 = sum(C_ni / W_ni^2) where C_ni = kurtosis factor = W(1-4W) ??
    # Actually Var(z^2) = (1/W - 4). 
    # Var(Outfit) = sum((1/W_ni - 4)) / N^2
    
    outfit_var = np.sum((1.0 / W_matrix) - 4.0, axis=0) / (n_persons**2)
    # Clip to avoid negative var in extreme edge cases (unlikely with items)
    outfit_var = np.maximum(outfit_var, 1e-10)
    outfit_t = (np.power(outfit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(outfit_var)) + (np.sqrt(outfit_var) / 3.0)
    
    # For Infit t:
    # Var(Infit) = sum( (W - 4W^2) ) / (sum(W))^2
    infit_var = np.sum(W_matrix - 4.0 * W_matrix**2, axis=0) / (np.sum(W_matrix, axis=0)**2)
    infit_var = np.maximum(infit_var, 1e-10)
    infit_t = (np.power(infit_msq, 1/3.0) - 1.0) * (3.0 / np.sqrt(infit_var)) + (np.sqrt(infit_var) / 3.0)
    
    # Combine results
    results = pd.DataFrame({
        'ItemIndex': np.arange(1, n_items + 1),
        'ItemName': item_cols,
        'Difficulty': item_difficulties,
        'SE': item_se,
        'InfitMNSQ': infit_msq,
        'OutfitMNSQ': outfit_msq,
        'Infit_t': infit_t,
        'Outfit_t': outfit_t,
        'PtMeasureCorr': pm_corrs
    })
    
    # Sort by Difficulty descending (largest first)
    results = results.sort_values(by='Difficulty', ascending=False)
    
    print("\nCalibration Results (First 5 items):")
    print(results.head())
    
    if person_id_df is not None:
        df_out = person_id_df.copy()
        df_out['RawScore'] = full_scores
        df_out['Measure'] = person_measures
    else:
        df_out = None
    
    return results, df_out

if __name__ == "__main__":
    items_df, persons_df = rasch_cmle()
    if items_df is not None:
        items_df.to_csv('rasch_item_params.csv', index=False)
        persons_df.to_csv('rasch_person_measures.csv', index=False)
        print("Saved rasch_item_params.csv and rasch_person_measures.csv")
