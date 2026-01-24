import pandas as pd
import numpy as np
import sys
import os

# Import read_blot_data
try:
    from readBLOT import read_blot_data
except ImportError:
    sys.path.append(os.getcwd())
    from readBLOT import read_blot_data

def solve_newton_mle(target_score, params, is_person=True, init_val=0.0):
    """
    Newton-Raphson solver for one parameter (theta or delta).
    target_score: Observed score (r_n or S_i)
    params: Array of fixed parameters (deltas if solving theta, thetas if solving delta)
    is_person: True if solving for theta (params=deltas), False if solving for delta (params=thetas)
               Logic: P = exp(theta - delta) / (1 + exp)
    """
    x = init_val
    for _ in range(20):
        if is_person:
            # Solving theta given deltas
            # logit = theta - deltas
            logits = x - params
        else:
            # Solving delta given thetas
            # logit = thetas - delta
            logits = params - x
            
        prob = np.exp(logits) / (1.0 + np.exp(logits))
        expected_score = np.sum(prob)
        variance = np.sum(prob * (1.0 - prob))
        
        if variance < 1e-9:
            break
            
        # Update: x_new = x_old + (Target - Exp) / Variance
        # For delta: dE/ddelta = -sum(P(1-P)). Change sign of update is handled by (Target - Exp) logic?
        # Let's trace:
        # If is_person: score function is Sum(P) - r = 0 ?? No, MLE is score - Sum(P) = 0.
        # f(t) = Sum(P) - r. f'(t) = Sum(P(1-P))
        # update = - f(t)/f'(t) = (r - Sum(P)) / Sum(P(1-P))
        
        # If is_item: P = exp(t-d)/(1+...). 
        # f(d) = Sum(P) - S. f'(d) = Sum( P(1-P) * (-1) ) = -Sum(P(1-P))
        # update = - (Sum(P) - S) / (-Var) = (S - Sum(P)) / -Var = (Sum(P) - S) / Var
        # Wait, usually delta update is (Expected - Observed) / Variance?
        
        diff = target_score - expected_score
        
        if is_person:
             step = diff / variance
             x += step
        else:
             # For item difficulty, if Exp > Obs, it means item is too easy (delta too low).
             # We need to increase delta.
             # diff = Obs - Exp. If Obs < Exp, diff is neg. Delta should increase? 
             # No, if Obs < Exp, item is harder than predicted. Increase diff.
             # Sign check:
             # P decreases as delta increases.
             # If we increase delta, Exp decreases.
             # If Obs < Exp, we want Exp to decrease. So increase delta.
             # step ~ (Exp - Obs) ?
             # diff = Obs - Exp. So step should be -diff.
             step = (expected_score - target_score) / variance
             x += step
             
        if np.abs(step) < 1e-4:
            break
    return x

def main():
    print("Loading Data...")
    df = read_blot_data('blot.txt')
    
    # Extract Matrix
    item_cols = [c for c in df.columns if c != 'PersonID']
    # Safety Check: Ensure item_cols matches data shape
    # If readBLOT created dataframe with mismatches, we fix it here.
    # We want columns corresponding to actual numeric data.
    # readBLOT guarantees df structure, but if columns > data, valid columns are first NI.
    
    # Re-infer NI from data values excluding ID
    # Actually, df.shape[1] - 1 is the number of item columns in the dataframe
    n_items_df = df.shape[1] - 1
    
    # If item_cols list is longer than n_items_df (e.g. duplicate names in header vs dataframe columns)
    if len(item_cols) > n_items_df:
        print(f"Warning: items in header ({len(item_cols)}) != dataframe columns ({n_items_df}). Truncating.")
        item_cols = item_cols[:n_items_df]

    # Re-extract proper data matrix
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Final check
    if X.shape[1] != len(item_cols):
         print(f"Error: X shape {X.shape} mismatch with label count {len(item_cols)}")
         # Force sync
         n_i = min(X.shape[1], len(item_cols))
         X = X[:, :n_i]
         item_cols = item_cols[:n_i]
    
    n_persons, n_items = X.shape
    print(f"JMLE Analysis: {n_persons} persons, {n_items} items")
    
    # Exclude perfect/zero scores
    person_scores = np.sum(X, axis=1)
    valid_mask = (person_scores > 0) & (person_scores < n_items)
    X_calib = X[valid_mask]
    ids_calib = df.loc[valid_mask, 'PersonID'].values
    scores_calib = person_scores[valid_mask]
    
    n_p_calib = len(X_calib)
    print(f"Calibration Sample: {n_p_calib}")
    
    # Initial Estimates (PROX-like or Log-odds)
    # Item Difficulties
    item_scores_calib = np.sum(X_calib, axis=0)
    # Simple logit of proportion
    p_i = item_scores_calib / n_p_calib
    p_i = np.clip(p_i, 0.01, 0.99)
    item_diffs = np.log((1 - p_i) / p_i)
    # Center Items initially
    item_diffs = item_diffs - np.mean(item_diffs)
    
    # Person Abilities
    # PROX initialization often better:
    # theta_n = log(r/(L-r))
    p_n = scores_calib / n_items
    p_n = np.clip(p_n, 0.01, 0.99)
    person_thetas = np.log(p_n / (1 - p_n))
    
    # JMLE Iteration
    max_iter = 100
    for it in range(max_iter):
        max_change = 0.0
        
        # 1. Update Persons (Theta) given Items (Deltas)
        # Standard Newton-Raphson update
        # Theta(new) = Theta(old) - (ModelScore - RawScore) / ModelVar
        # But commonly in Winsteps: theta += (Raw - Expected) / Variance
        
        # Calculate Expected Scores and Variances for all persons in one go using broadcasting
        # Matrix P [Persons x Items]
        # P_ni = exp(theta_n - delta_i) / (1 + exp)
        
        # Broadcasting: theta (n,1) - delta (1,i) = (n,i) matrix
        logit_mat = person_thetas[:, np.newaxis] - item_diffs[np.newaxis, :]
        P_mat = np.exp(logit_mat) / (1.0 + np.exp(logit_mat))
        
        row_expected = np.sum(P_mat, axis=1)
        row_variance = np.sum(P_mat * (1.0 - P_mat), axis=1)
        
        # Avoid zero variance division
        row_variance = np.maximum(row_variance, 1e-10)
        
        dp = (scores_calib - row_expected) / row_variance
        
        # Clip update to avoid oscillation
        dp = np.clip(dp, -1.0, 1.0)
        
        person_thetas += dp
        change_p = np.max(np.abs(dp))
        
        # 2. Update Items (Delta) given Persons (Thetas)
        # Recalculate P_mat with new thetas? Or just use previous? 
        # JMLE usually alternate. Let's recalculate Prob matrix for accuracy.
        
        logit_mat = person_thetas[:, np.newaxis] - item_diffs[np.newaxis, :]
        P_mat = np.exp(logit_mat) / (1.0 + np.exp(logit_mat))
        
        col_expected = np.sum(P_mat, axis=0)
        col_variance = np.sum(P_mat * (1.0 - P_mat), axis=0)
        
        col_variance = np.maximum(col_variance, 1e-10)
        
        # Score_i = Sum(P_ni) - R_i = 0
        # Update delta: delta_new = delta_old - (Expected - Observed) / (-Variance)
        # = delta_old + (Expected - Observed) / Variance  <-- Wait.
        # Let's check sign.
        # Prob P = e(t-d)/(1+e). dP/dd = -P(1-P)
        # Func f(d) = Expected(d) - Observed. We want f(d)=0.
        # f'(d) = Sum(dP/dd) = -Sum P(1-P) = -Variance
        # Newton: d_new = d - f(d)/f'(d) = d - (Exp - Obs)/(-Var) = d + (Exp - Obs)/Var
        # If Exp > Obs (predicting too high score), item is harder -> delta should increase.
        # If Exp > Obs, term is positive. Delta increases. Correct.
        
        dd = (col_expected - item_scores_calib) / col_variance
        dd = np.clip(dd, -1.0, 1.0)
        
        item_diffs += dd
        
        # Re-center items to mean 0 immediately
        mean_diff = np.mean(item_diffs)
        item_diffs -= mean_diff
        # Important: In JMLE, shifting items requires shifting persons to maintain logits
        # BUT many implementations just fix item mean=0 at every step effectively.
        # PROPER WAY: Shift items, shift persons same amount.
        person_thetas -= mean_diff 
        # But wait, we want sum(deltas)=0 constraint.
        # If we subtract M from deltas, we effectively added (t - (d-M)) = t - d + M.
        # To keep (t-d) constant, we must subtract M from t as well.
        # Yes.
        
        change_i = np.max(np.abs(dd))
        
        total_change = max(change_p, change_i)
        # print(f"Iter {it}: Change {total_change:.6f}")
        
        if total_change < 0.005: # Winsteps default often .01 or .005
            print(f"Converged at iter {it}")
            break
            
    # Winsteps Bias Correction for JMLE UCON
    # Factor (L-1)/L applied to BOTH item and person spreads?
    # Actually, Winsteps UCON method effectively multiplies logits by (L-1)/L to correct for bias.
    # UCON bias correction: L/(L-1) is applied to variance? No.
    # Wright & Stone: Item measures multiplied by (L-1)/L.
    
    L = n_items
    correction_factor = (L - 1) / L
    
    print(f"Applying Bias Correction factor {correction_factor:.4f}")
    item_diffs_corrected = item_diffs * correction_factor
    person_thetas_corrected = person_thetas * correction_factor
    
    # Calculate Fit Statistics (based on corrected params)
    print("Computing Fit Statistics...")
    
    P_matrix = np.zeros((n_p_calib, n_items))
    for n in range(n_p_calib):
        # P = exp(theta - delta) / (1+exp)
        logit = person_thetas_corrected[n] - item_diffs_corrected
        P_matrix[n, :] = np.exp(logit) / (1.0 + np.exp(logit))
        
    W_matrix = P_matrix * (1.0 - P_matrix)
    Residuals = X_calib - P_matrix
    StdRes = Residuals / np.sqrt(W_matrix)
    
    # Infit/Outfit
    infit_msq = np.sum(Residuals**2, axis=0) / np.sum(W_matrix, axis=0)
    outfit_msq = np.sum(StdRes**2, axis=0) / n_p_calib
    
    # Standard Error
    # SE(delta) = 1 / sqrt(sum(W))
    item_se = 1.0 / np.sqrt(np.sum(W_matrix, axis=0))
    
    # Point Measure Correlation
    pm_corrs = []
    for i in range(n_items):
        # Corr between obs (X column) and person measures
        # Use only calibration sample
        r = np.corrcoef(X_calib[:, i], person_thetas_corrected)[0, 1]
        pm_corrs.append(r)
        
    results = pd.DataFrame({
        'ItemIndex': np.arange(1, n_items + 1),
        'ItemName': item_cols,
        'Difficulty': item_diffs_corrected,
        'SE': item_se,
        'InfitMNSQ': infit_msq,
        'OutfitMNSQ': outfit_msq,
        'PtMeasureCorr': pm_corrs
    })
    
    results = results.sort_values(by='Difficulty', ascending=False)
    
    print("\nJMLE Calibration Results (First 5 items):")
    print(results.head())
    
    results.to_csv('rasch_jmle_items.csv', index=False)
    print("Saved to rasch_jmle_items.csv")

if __name__ == "__main__":
    main()
