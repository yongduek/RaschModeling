import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import os
import sys

# Import local modules
try:
    from readBLOT import read_blot_data
    from rasch_cmle import estimate_rasch_cmle_from_matrix
except ImportError:
    sys.path.append(os.getcwd())
    from readBLOT import read_blot_data
    from rasch_cmle import estimate_rasch_cmle_from_matrix

def run_split_sample_analysis(datafile='blot.txt'):
    print(f"Reading data from {datafile}...")
    df = read_blot_data(datafile)
    if df is None or df.empty:
        print("Error: Could not read data.")
        return

    # Extract binary matrix
    # Assuming first col is ID
    item_cols = df.columns[1:]
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    n_persons = len(X)
    print(f"Total N: {n_persons}")
    
    # Sort by Ability (Raw Score is sufficient statistic for Person Measure in Rasch)
    # Higher score = Higher ability
    raw_scores = np.sum(X, axis=1)
    
    # Get indices that would sort the array
    # argsort sorts ascending (low to high)
    sorted_indices = np.argsort(raw_scores)
    
    # Split into High and Low groups
    # Low group: First 75 (lowest scores)
    # High group: Last 75 (highest scores)
    cutoff = n_persons // 2
    
    idx_low = sorted_indices[:cutoff]
    idx_high = sorted_indices[cutoff:]
    
    X_low = X[idx_low]
    X_high = X[idx_high]
    
    print(f"Split by Ability:")
    print(f" - Low Ability Group (n={len(X_low)}), Avg Score: {np.mean(raw_scores[idx_low]):.2f}")
    print(f" - High Ability Group (n={len(X_high)}), Avg Score: {np.mean(raw_scores[idx_high]):.2f}")
    
    # Estimate Low Group
    print("Estimating Low Ability Group...")
    res_low, _ = estimate_rasch_cmle_from_matrix(X_low, item_cols=item_cols)
    if res_low is None:
        print("Estimation failed for Low Group")
        return
    # Ensure correct order (sort by ItemIndex to align with other group)
    res_low = res_low.sort_values('ItemIndex')
    diff_low = res_low['Difficulty'].values
    se_low = res_low['SE'].values

    # Estimate High Group
    print("Estimating High Ability Group...")
    res_high, _ = estimate_rasch_cmle_from_matrix(X_high, item_cols=item_cols)
    if res_high is None:
        print("Estimation failed for High Group")
        return
    res_high = res_high.sort_values('ItemIndex')
    diff_high = res_high['Difficulty'].values
    se_high = res_high['SE'].values
        
    # Analysis
    corr, _ = pearsonr(diff_low, diff_high)
    print(f"\nItem Difficulty Correlation (Low vs High): {corr:.4f}")
    
    # Create DataFrame for results
    res_df = pd.DataFrame({
        'Item': res_low['ItemName'],
        'Diff_Low': diff_low,
        'SE_Low': se_low,
        'Diff_High': diff_high,
        'SE_High': se_high,
        'Diff_Diff': diff_low - diff_high
    })
    
    print("\nTop 5 Items by Absolute Difference:")
    res_df['Abs_Diff'] = res_df['Diff_Diff'].abs()
    print(res_df.sort_values('Abs_Diff', ascending=False).head())
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    # X-axis: High Ability, Y-axis: Low Ability
    
    # 95% Confidence Ellipses (joint probability)
    # Chi-square critical value for 2df at 95% is 5.991
    # Scale factor = sqrt(5.991) approx 2.4477
    chisq_scale = np.sqrt(5.991)
    
    for i in range(len(diff_low)):
        # Width and Height are total lengths (diameter), so 2 * scale * SE
        width = 2 * chisq_scale * se_high[i]
        height = 2 * chisq_scale * se_low[i]
        el = Ellipse(xy=(diff_high[i], diff_low[i]), width=width, height=height, 
                     edgecolor='blue', facecolor='blue', alpha=0.1)
        ax.add_patch(el)
        
    ax.scatter(diff_high, diff_low, c='blue', alpha=0.6, label='Items', s=20)
    
    # Label items
    for i in range(len(diff_low)):
        ax.annotate(str(i + 1), (diff_high[i], diff_low[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)

    # Add identity line
    min_val = min(min(diff_low), min(diff_high)) - 1.5
    max_val = max(max(diff_low), max(diff_high)) + 1.5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Identity Line')
    
    # 95% Control Lines (Classical Invariance Bands)
    # Visualizes item bias relative to the diagonal
    sorted_idx = np.argsort(diff_high)
    sorted_x = diff_high[sorted_idx]
    
    # Calculate SE of difference for each item
    se_diff = np.sqrt(se_low**2 + se_high**2)
    sorted_se_diff = se_diff[sorted_idx]
    
    upper_curve = sorted_x + 1.96 * sorted_se_diff
    lower_curve = sorted_x - 1.96 * sorted_se_diff
    
    ax.plot(sorted_x, upper_curve, 'g:', alpha=0.8, label='95% Control Lines')
    ax.plot(sorted_x, lower_curve, 'g:', alpha=0.8)

    ax.set_title(f'Item Parameter Invariance (Ability Split)\nr = {corr:.3f}')
    ax.set_xlabel(f'High Ability Group Difficulty (n={len(X_high)})')
    ax.set_ylabel(f'Low Ability Group Difficulty (n={len(X_low)})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set limits explicitly to show full ellipses
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    out_img = 'split_sample_plot.png'
    plt.savefig(out_img)
    print(f"\nPlot saved to {out_img}")
    
    # Save CSV
    res_df.to_csv('split_sample_results.csv', index=False)
    print("Results saved to split_sample_results.csv")

if __name__ == '__main__':
    run_split_sample_analysis()
