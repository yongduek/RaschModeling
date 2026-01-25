import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, Any
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def plot_icc(target_id: Union[int, str],
             items_df: pd.DataFrame,
             persons_df: pd.DataFrame,
             raw_df: pd.DataFrame,
             color: str = 'blue',
             show: bool = True,
             compute_slope: bool = False) -> Dict[str, Any]:
    """
    Plots the Empirical vs Model ICC for a given item.

    Explanation of Plot Elements:
    -----------------------------
    The orange circles represent the ACTUAL OBSERVED performance of students, grouped by ability.
    - Each Circle: Represents a group of students who all share the exact same Ability Measure (Total Score).
    - Y-Position (Height): The Proportion Correct for that specific group (e.g., 0.8 means 80% of students with that score got the item correct).
    - X-Position (Horizontal): The estimated Ability Measure (in logits) for that group.
    - Size of the Circle: Represents the Sample Size (N). Larger circles mean more students had that score, making the data point more reliable. Tiny circles represent fewer students (more noise).

    Interpretation:
    - If the circles closely follow the blue curve, it indicates that the data fits the Rasch Model well for that item.
    - Good Fit: Circles are near the curve, suggesting the model accurately predicts student performance.
    - Misfit: If valid-sized orange circles deviate significantly from the blue line, it means students of that ability performed differently on this item than the Rasch model predicted.
    
    Parameters:
    -----------
    target_id : int or str
        The ItemIndex (int) or a substring of the ItemName (str) to identify the item.
    items_df : pd.DataFrame
        DataFrame containing item statistics (must have 'ItemName', 'Difficulty', and optionally 'ItemIndex').
    persons_df : pd.DataFrame
        DataFrame containing person statistics (must have 'Measure').
    raw_df : pd.DataFrame
        DataFrame containing raw item responses (0/1).
    color : str
        Color for the theoretical model curve.
    show : bool
        If True, show the matplotlib figure. If False, suppress plotting output.
    compute_slope : bool
        If True, estimate an empirical slope via a 2PL fit and return metrics.
    """
    # 1. Identify Item
    item_row = pd.DataFrame()
    
    # Try integer index match first (assuming ItemIndex column exists and matches target_id)
    if isinstance(target_id, int) and 'ItemIndex' in items_df.columns:
        item_row = items_df[items_df['ItemIndex'] == target_id]
        
    # If not found or target is string, try name matching
    if item_row.empty:
        s_target = str(target_id)
        # Regex to match "ID_Name" or just "ID" strictly at start
        # We use a raw string for regex
        pat = fr"^{s_target}\D|^{s_target}$"
        item_row = items_df[items_df['ItemName'].str.match(pat)]
        
        # Fallback to simple contains
        if item_row.empty:
             item_row = items_df[items_df['ItemName'].str.contains(s_target)]
    
    if item_row.empty:
        print(f"Item '{target_id}' not found.")
        return

    # Extract Details
    item_row = item_row.iloc[0]
    item_name = item_row['ItemName']
    item_diff = item_row['Difficulty']
    result: Dict[str, Any] = {
        'item_name': item_name,
        'difficulty': float(item_diff)
    }
    print(f"Target Item: {item_name}, Difficulty (Delta) = {item_diff:.3f}")

    # 2. Prepare Data for Empirical Points
    # Match person measures with their response
    plot_df = pd.DataFrame({
        'Measure': persons_df['Measure'],
        'Response': raw_df[item_name]
    })
    
    # Filter out undefined measures
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Group by Measure
    empirical_points = plot_df.groupby('Measure')['Response'].agg(['mean', 'count']).reset_index()
    empirical_points.columns = ['Measure', 'Proportion', 'N']

    # 3. Generate Theoretical Curve
    theta_min, theta_max = plot_df['Measure'].min(), plot_df['Measure'].max()
    theta_range = np.linspace(theta_min - 1.0, theta_max + 1.0, 100)
    logits = theta_range - item_diff
    theoretical_prob = np.exp(logits) / (1 + np.exp(logits))

    # Optional empirical slope estimation
    if compute_slope and _HAS_SCIPY:
        def two_param_logistic(theta, a, b):
            return 1 / (1 + np.exp(-a * (theta - b)))
        try:
            popt, _ = curve_fit(two_param_logistic,
                                plot_df['Measure'].values,
                                plot_df['Response'].values,
                                p0=[1.0, float(item_diff)],
                                maxfev=10000)
            result['empirical_slope'] = float(popt[0])
            result['empirical_diff'] = float(popt[1])
        except Exception:
            result['empirical_slope'] = None
            result['empirical_diff'] = None

    # 4. Plot
    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, theoretical_prob, label=f'Model ICC (Delta={item_diff:.2f})', color=color, linewidth=2)
        sns.scatterplot(data=empirical_points, x='Measure', y='Proportion', size='N', sizes=(20, 200),
                        color='orange', alpha=0.8, edgecolor='black', zorder=5)

        plt.title(f'Item Characteristic Curve: {item_name}', fontsize=15)
        plt.xlabel('Person Ability (Logits)')
        plt.ylabel('Probability of Correct Response')
        plt.axvline(item_diff, color='red', linestyle=':', label='Item Difficulty')
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

    return result
