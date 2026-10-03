from .preprocessing import preprocess_baseline, preprocess_proposed
from .model import train_model, evaluate_model
import numpy as np

def run_experiment(train_df, val_df, test_df, missing_frac=0.0):
    # Baseline
    X_train_b, y_train_b = preprocess_baseline(train_df)
    X_val_b, y_val_b = preprocess_baseline(val_df)
    X_test_b, y_test_b = preprocess_baseline(test_df)
    
    # Proposed
    X_train_p, y_train_p = preprocess_proposed(train_df)
    X_val_p, y_val_p = preprocess_proposed(val_df)
    X_test_p, y_test_p = preprocess_proposed(test_df)
    
    # Assertions for fair comparison on Test Set
    # We drop the index because concatenation might have reset them differently
    # But subject_id and charttime must match exactly.
    assert len(y_test_b) == len(y_test_p), "Mismatch in evaluation sample counts!"
    
    # Align by index assuming they are processed in the same order
    mismatched_timestamps = (X_test_b['charttime'].values != X_test_p['charttime'].values).sum()
    max_diff_target = np.max(np.abs(y_test_b.values - y_test_p.values))
    
    assert mismatched_timestamps == 0, "Mismatch in target timestamps!"
    assert max_diff_target < 1e-6, "Mismatch in target values!"
    
    # We only print this once when the experiment is manually run or when running via print (or we can just print it always)
    print(f"\n--- Validation Check (Missingness: {missing_frac}) ---")
    print(f"Number of common evaluation samples: {len(y_test_b)}")
    print(f"Maximum absolute difference between target values: {max_diff_target}")
    print(f"Number of mismatched target timestamps: {mismatched_timestamps}")
    
    # Remove identifier columns for training
    features_b = [col for col in X_train_b.columns if col not in ['subject_id', 'charttime']]
    features_p = [col for col in X_train_p.columns if col not in ['subject_id', 'charttime']]
    
    # Train
    model_b = train_model(X_train_b[features_b], y_train_b)
    model_p = train_model(X_train_p[features_p], y_train_p)
    
    # Evaluate
    metrics_b = evaluate_model(model_b, X_test_b[features_b], y_test_b)
    metrics_p = evaluate_model(model_p, X_test_p[features_p], y_test_p)
    
    mae_reduction = 0
    if metrics_b["MAE"] > 0 and not __import__('math').isnan(metrics_b["MAE"]):
        mae_reduction = (metrics_b["MAE"] - metrics_p["MAE"]) / metrics_b["MAE"] * 100
        
    return {
        "Baseline": metrics_b,
        "Proposed": metrics_p,
        "MAE_Reduction_Pct": mae_reduction
    }
