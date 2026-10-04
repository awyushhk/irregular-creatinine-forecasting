import numpy as np
from .preprocessing import preprocess_pipeline_A, preprocess_pipeline_B, preprocess_pipeline_C
from .model import train_model, evaluate_model

def run_experiment(train_df, val_df, test_df, missing_frac=0.0):
    X_train_a, y_train_a = preprocess_pipeline_A(train_df)
    X_val_a, y_val_a = preprocess_pipeline_A(val_df)
    X_test_a, y_test_a = preprocess_pipeline_A(test_df)

    X_train_b, y_train_b = preprocess_pipeline_B(train_df)
    X_val_b, y_val_b = preprocess_pipeline_B(val_df)
    X_test_b, y_test_b = preprocess_pipeline_B(test_df)
    
    X_train_c, y_train_c = preprocess_pipeline_C(train_df)
    X_val_c, y_val_c = preprocess_pipeline_C(val_df)
    X_test_c, y_test_c = preprocess_pipeline_C(test_df)
    
    # Assertions
    assert len(y_test_a) == len(y_test_b) == len(y_test_c), "Mismatch in sample counts!"
    
    assert (X_test_a['charttime'].values == X_test_b['charttime'].values).all(), "A/B timestamp mismatch"
    assert (X_test_b['charttime'].values == X_test_c['charttime'].values).all(), "B/C timestamp mismatch"
    assert (X_test_a['subject_id'].values == X_test_b['subject_id'].values).all(), "A/B subject mismatch"
    
    assert np.max(np.abs(y_test_a.values - y_test_b.values)) < 1e-6, "A/B target mismatch"
    assert np.max(np.abs(y_test_b.values - y_test_c.values)) < 1e-6, "B/C target mismatch"
    
    features_a = [col for col in X_train_a.columns if col not in ['subject_id', 'charttime']]
    features_b = [col for col in X_train_b.columns if col not in ['subject_id', 'charttime']]
    features_c = [col for col in X_train_c.columns if col not in ['subject_id', 'charttime']]
    
    model_a = train_model(X_train_a[features_a], y_train_a)
    model_b = train_model(X_train_b[features_b], y_train_b)
    model_c = train_model(X_train_c[features_c], y_train_c)
    
    metrics_a = evaluate_model(model_a, X_test_a[features_a], y_test_a)
    metrics_b = evaluate_model(model_b, X_test_b[features_b], y_test_b)
    metrics_c = evaluate_model(model_c, X_test_c[features_c], y_test_c)
    
    return {
        "Pipeline A": metrics_a,
        "Pipeline B": metrics_b,
        "Pipeline C": metrics_c,
        "n_samples": len(y_test_a)
    }
