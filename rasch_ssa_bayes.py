import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import pearsonr
import os
import sys

# Import local modules
try:
    from readBLOT import read_blot_data
except ImportError:
    sys.path.append(os.getcwd())
    from readBLOT import read_blot_data

def estimate_rasch_bayes(X, item_cols=None):
    """
    Estimates Rasch parameters using CmdStanPy.
    X: Binary matrix (n_persons, n_items). NaNs treated as missing.
    """
    if item_cols is None:
        item_cols = [f'Item_{i+1}' for i in range(X.shape[1])]
    
    n_persons, n_items = X.shape
    
    # Needs long format
    # Create simple dataframe
    df_X = pd.DataFrame(X, columns=item_cols)
    df_X['PersonIdx'] = np.arange(1, n_persons + 1)
    
    df_long = df_X.melt(id_vars='PersonIdx', var_name='Item', value_name='Response')
    
    # Drop missing
    df_long['Response'] = pd.to_numeric(df_long['Response'], errors='coerce')
    df_long = df_long.dropna(subset=['Response'])
    df_long['Response'] = df_long['Response'].astype(int)
    
    # Map Items to 1..K
    # Ensure consistent order
    item_map = {name: i+1 for i, name in enumerate(item_cols)}
    df_long['ItemIdx'] = df_long['Item'].map(item_map)
    
    stan_data = {
        'N_obs': len(df_long),
        'N_persons': n_persons,
        'N_items': n_items,
        'jj': df_long['PersonIdx'].values,
        'kk': df_long['ItemIdx'].values,
        'y': df_long['Response'].values
    }
    
    # Compile
    stan_file = 'rasch.stan'
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    
    # Sample (Using optimization or shorter chain for speed in this context? 
    # Usually Bayesian implies full MCMC. Let's do reasonable short chains.)
    # Note: For split-sample, we need two runs. Speed matters.
    # 500 warmup, 500 samples gives decent estimates for mean.
    fit = model.sample(data=stan_data, chains=4, iter_warmup=500, iter_sampling=500, show_progress=False)
    
    summary = fit.summary()
    
    # Extract Deltas
    delta_rows = [idx for idx in summary.index if idx.startswith('delta[')]
    item_results = summary.loc[delta_rows, ['Mean', 'StdDev']]
    
    item_results['ItemName'] = item_cols
    item_results['ItemIndex'] = range(1, n_items + 1)
    item_results = item_results.rename(columns={'Mean': 'Difficulty', 'StdDev': 'SE'})
    
    return item_results

def run_split_sample_bayes(datafile='blot.txt'):
    print(f"Reading data from {datafile}...")
    df = read_blot_data(datafile)
    if df is None or df.empty: return

    item_cols = df.columns[1:]
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    n_persons = len(X)
    raw_scores = np.sum(X, axis=1)
    sorted_indices = np.argsort(raw_scores)
    
    cutoff = n_persons // 2
    idx_low = sorted_indices[:cutoff]
    idx_high = sorted_indices[cutoff:]
    
    X_low = X[idx_low]
    X_high = X[idx_high]
    
    print(f"Split N={n_persons} into Low (n={len(X_low)}) and High (n={len(X_high)})")
    
    print("Estimating Low Group (Bayes)...")
    res_low = estimate_rasch_bayes(X_low, item_cols)
    diff_low = res_low['Difficulty'].values
    se_low = res_low['SE'].values
    
    print("Estimating High Group (Bayes)...")
    res_high = estimate_rasch_bayes(X_high, item_cols)
    diff_high = res_high['Difficulty'].values
    se_high = res_high['SE'].values
    
    corr, _ = pearsonr(diff_low, diff_high)
    print(f"\nCorrection: {corr:.4f}")
    
    # Save Results
    res_df = pd.DataFrame({
        'Item': item_cols,
        'Diff_Low': diff_low,
        'SE_Low': se_low,
        'Diff_High': diff_high,
        'SE_High': se_high,
        'Diff_Diff': diff_low - diff_high
    })
    res_df.to_csv('split_sample_bayes.csv', index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    chisq_scale = np.sqrt(5.991)
    
    for i in range(len(diff_low)):
        width = 2 * chisq_scale * se_high[i]
        height = 2 * chisq_scale * se_low[i]
        el = Ellipse(xy=(diff_high[i], diff_low[i]), width=width, height=height, 
                     edgecolor='blue', facecolor='blue', alpha=0.1)
        ax.add_patch(el)
        
    ax.scatter(diff_high, diff_low, c='blue', alpha=0.6, s=20)
    for i in range(len(diff_low)):
        ax.annotate(str(i + 1), (diff_high[i], diff_low[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)

    min_val = min(min(diff_low), min(diff_high)) - 1.5
    max_val = max(max(diff_low), max(diff_high)) + 1.5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Control Lines
    sorted_idx = np.argsort(diff_high)
    sorted_x = diff_high[sorted_idx]
    se_diff = np.sqrt(se_low**2 + se_high**2)
    sorted_se_diff = se_diff[sorted_idx]
    ax.plot(sorted_x, sorted_x + 1.96*sorted_se_diff, 'g:', alpha=0.5)
    ax.plot(sorted_x, sorted_x - 1.96*sorted_se_diff, 'g:', alpha=0.5)
    
    ax.set_title(f'Bayesian Split-Sample (Ability)\nr = {corr:.3f}')
    ax.set_xlabel('High Ability (Bayes Logits)')
    ax.set_ylabel('Low Ability (Bayes Logits)')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, alpha=0.3)
    
    plt.savefig('split_sample_bayes_plot.png')
    print("Saved plot to split_sample_bayes_plot.png")

if __name__ == '__main__':
    run_split_sample_bayes()
