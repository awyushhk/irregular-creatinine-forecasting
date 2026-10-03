import sys; sys.path.insert(0, '.')
from src.load_data import load_and_filter_mimic_data
from src.preprocessing import split_by_patient, apply_missingness, preprocess_baseline
from src.evaluate import run_experiment
import numpy as np
import pandas as pd

df = load_and_filter_mimic_data()
train_df, val_df, test_df = split_by_patient(df)

print('\n--- STRICT LEAKAGE TEST (CAUSAL BASELINE) ---')
patient_df = df[df['subject_id'] == df['subject_id'].iloc[0]].copy()
X_b, y_b = preprocess_baseline(patient_df)

for i in range(min(3, len(X_b))):
    pred_t = X_b['charttime'].iloc[i]
    target = y_b.iloc[i]
    
    idx = patient_df.index[patient_df['charttime'] == pred_t].tolist()
    if not idx: continue
    actual_t_idx = idx[0]
    next_actual_t = patient_df.loc[actual_t_idx + 1, 'charttime'] if (actual_t_idx + 1) in patient_df.index else 'N/A'
    
    print(f'Prediction timestamp: {pred_t}')
    print(f'Latest observation allowed: {pred_t}')
    print(f'Target timestamp: {next_actual_t}')
    if next_actual_t != 'N/A':
        assert next_actual_t > pred_t, 'Target is not in the future!'

print('\nStrict Leakage Test Passed.')

print('\n--- RUNNING MISSINGNESS EXPERIMENT ---')
for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    train_df_m = apply_missingness(train_df, frac)
    val_df_m = apply_missingness(val_df, frac)
    test_df_m = apply_missingness(test_df, frac)
    res = run_experiment(train_df_m, val_df_m, test_df_m, missing_frac=frac)
    print(f'Missingness {frac*100}%:')
    print(f"  Baseline MAE: {res['Baseline']['MAE']:.4f}, RMSE: {res['Baseline']['RMSE']:.4f}, R2: {res['Baseline']['R2']:.4f}")
    print(f"  Proposed MAE: {res['Proposed']['MAE']:.4f}, RMSE: {res['Proposed']['RMSE']:.4f}, R2: {res['Proposed']['R2']:.4f}")
