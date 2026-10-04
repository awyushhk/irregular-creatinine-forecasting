import pandas as pd
import numpy as np

def preprocess_pipeline_A(df):
    df = df.copy()
    df = df.sort_values(['subject_id', 'charttime'])
    processed_dfs = []
    
    for subject_id, group in df.groupby('subject_id'):
        group = group.copy()
        group['target'] = group['valuenum'].shift(-1)
        group['is_valid'] = group['target'].notna() & group['valuenum'].shift(1).notna()
        group['date'] = group['charttime'].dt.floor('D')
        
        feature_rows = []
        for i in range(len(group)):
            row = group.iloc[i]
            t = row['charttime']
            history = group.iloc[:i+1]
            daily_means = history.groupby('date')['valuenum'].mean()
            
            if len(daily_means) > 0:
                min_date = daily_means.index.min()
                current_date = t.floor('D')
                full_index = pd.date_range(min_date, current_date, freq='D')
                daily_grid = daily_means.reindex(full_index)
                
                # Constrained interpolation: only gaps <= 7 days
                mask = daily_grid.isna()
                cumsum = (~mask).cumsum()
                gap_sizes = mask.groupby(cumsum).transform('sum')
                
                daily_interp = daily_grid.interpolate(method='linear')
                daily_interp[mask & (gap_sizes > 7)] = np.nan
                daily_grid = daily_interp.ffill()
                
                current_creat = daily_grid.iloc[-1]
                prev_creat = daily_grid.iloc[-2] if len(daily_grid) >= 2 else current_creat
                last_3 = daily_grid.iloc[-3:]
                r_mean = last_3.mean()
                r_std = last_3.std() if len(last_3) >= 2 else 0.0
                if pd.isna(r_std): r_std = 0.0
            else:
                current_creat, prev_creat, r_mean, r_std = np.nan, np.nan, np.nan, 0.0
                
            feature_rows.append({
                'current_creatinine': current_creat,
                'previous_creatinine': prev_creat,
                'creatinine_change': current_creat - prev_creat,
                'rolling_mean': r_mean,
                'rolling_std': r_std
            })
            
        features_df = pd.DataFrame(feature_rows, index=group.index)
        merged = pd.concat([group, features_df], axis=1)
        processed_dfs.append(merged[merged['is_valid']].copy())
        
    if not processed_dfs:
        return pd.DataFrame(), pd.Series()
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    features = ['current_creatinine', 'previous_creatinine', 'creatinine_change', 'rolling_mean', 'rolling_std']
    return final_df[['subject_id', 'charttime'] + features], final_df['target']

def preprocess_pipeline_B(df):
    df = df.copy()
    df = df.sort_values(['subject_id', 'charttime'])
    processed_dfs = []
    
    for subject_id, group in df.groupby('subject_id'):
        group = group.copy()
        group['target'] = group['valuenum'].shift(-1)
        group['is_valid'] = group['target'].notna() & group['valuenum'].shift(1).notna()
        group['date'] = group['charttime'].dt.floor('D')
        
        feature_rows = []
        for i in range(len(group)):
            row = group.iloc[i]
            t = row['charttime']
            history = group.iloc[:i+1]
            daily_means = history.groupby('date')['valuenum'].mean()
            
            if len(daily_means) > 0:
                min_date = daily_means.index.min()
                current_date = t.floor('D')
                full_index = pd.date_range(min_date, current_date, freq='D')
                daily_grid = daily_means.reindex(full_index).ffill()
                
                current_creat = daily_grid.iloc[-1]
                prev_creat = daily_grid.iloc[-2] if len(daily_grid) >= 2 else current_creat
                last_3 = daily_grid.iloc[-3:]
                r_mean = last_3.mean()
                r_std = last_3.std() if len(last_3) >= 2 else 0.0
                if pd.isna(r_std): r_std = 0.0
            else:
                current_creat, prev_creat, r_mean, r_std = np.nan, np.nan, np.nan, 0.0
                
            feature_rows.append({
                'current_creatinine': current_creat,
                'previous_creatinine': prev_creat,
                'creatinine_change': current_creat - prev_creat,
                'rolling_mean': r_mean,
                'rolling_std': r_std
            })
            
        features_df = pd.DataFrame(feature_rows, index=group.index)
        merged = pd.concat([group, features_df], axis=1)
        processed_dfs.append(merged[merged['is_valid']].copy())
        
    if not processed_dfs:
        return pd.DataFrame(), pd.Series()
    final_df = pd.concat(processed_dfs, ignore_index=True)
    features = ['current_creatinine', 'previous_creatinine', 'creatinine_change', 'rolling_mean', 'rolling_std']
    return final_df[['subject_id', 'charttime'] + features], final_df['target']

def preprocess_pipeline_C(df):
    # Pipeline C - Reduced, Ablation-Justified (v2)
    # Features retained per ablation study: current_creatinine, previous_creatinine,
    # creatinine_change, delta_t_hours, rolling_mean, rolling_std.
    # Removed: observation_mask (constant=1.0, zero discriminative power),
    #          observation_count, observation_density, time_since_last_observation
    #          (duplicate of delta_t), observation_confidence.
    df = df.copy()
    processed_dfs = []
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('charttime').copy()

        group['target'] = group['valuenum'].shift(-1)
        group['is_valid'] = group['target'].notna() & group['valuenum'].shift(1).notna()

        group['current_creatinine'] = group['valuenum']
        group['previous_creatinine'] = group['current_creatinine'].shift(1)
        group['creatinine_change'] = group['current_creatinine'] - group['previous_creatinine']

        # Strictly causal: delta_t uses shift(1) charttime only
        group['prev_charttime'] = group['charttime'].shift(1)
        group['delta_t_hours'] = (
            group['charttime'] - group['prev_charttime']
        ).dt.total_seconds() / 3600.0
        group['delta_t_hours'] = group['delta_t_hours'].fillna(0.0)

        # Rolling features over exact observation sequence (causal, window=3)
        group['rolling_mean'] = group['current_creatinine'].rolling(window=3, min_periods=1).mean()
        group['rolling_std'] = group['current_creatinine'].rolling(window=3, min_periods=2).std().fillna(0.0)

        valid = group[group['is_valid']].copy()
        processed_dfs.append(valid)

    if not processed_dfs:
        return pd.DataFrame(), pd.Series()

    final_df = pd.concat(processed_dfs, ignore_index=True)
    features = [
        'current_creatinine', 'previous_creatinine', 'creatinine_change',
        'delta_t_hours', 'rolling_mean', 'rolling_std'
    ]
    return final_df[['subject_id', 'charttime'] + features], final_df['target']

# Keep split_by_patient and apply_missingness intact
def split_by_patient(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    np.random.seed(seed)
    patients = df['subject_id'].unique()
    np.random.shuffle(patients)
    n = len(patients)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]
    train_df = df[df['subject_id'].isin(train_patients)].copy()
    val_df = df[df['subject_id'].isin(val_patients)].copy()
    test_df = df[df['subject_id'].isin(test_patients)].copy()
    return train_df, val_df, test_df

def apply_missingness(df, missing_frac, seed=42):
    if missing_frac <= 0:
        return df.copy()
    np.random.seed(seed)
    indices_to_drop = []
    for subject_id, group in df.groupby('subject_id'):
        n = len(group)
        if n > 2:
            drop_count = int(n * missing_frac)
            if drop_count > 0:
                dropped = np.random.choice(group.index[1:-1], drop_count, replace=False)
                indices_to_drop.extend(dropped)
    return df.drop(index=indices_to_drop).copy()
