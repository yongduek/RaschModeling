import pandas as pd
import numpy as np
from readBLOT import read_blot_data

def check_item_stats(datafile='blot.txt'):
    # Load Data
    df = read_blot_data(datafile)
    item_cols = df.columns[1:]
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Calculate Total Scores and Split
    n_persons = len(X)
    raw_scores = np.sum(X, axis=1)
    sorted_indices = np.argsort(raw_scores)
    
    cutoff = n_persons // 2
    idx_high = sorted_indices[cutoff:]
    X_high = X[idx_high]
    
    print(f"High Ability Group N={len(X_high)}")
    
    # Items of interest: 7 and 12 (Indices 6 and 11, assuming 0-based)
    # Check column names
    target_indices = [6, 11]
    
    print("\nStatistics for Target Items in High Ability Group:")
    for i in target_indices:
        item_name = item_cols[i]
        scores = X_high[:, i]
        n_correct = np.sum(scores)
        n_total = len(scores)
        p_val = n_correct / n_total
        print(f"Item {i+1} ({item_name}): Correct={n_correct}/{n_total} (p={p_val:.3f})")

if __name__ == "__main__":
    check_item_stats()