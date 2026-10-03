import pandas as pd
import numpy as np

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

def preprocess_baseline(df):
    """
    Pipeline A: Baseline (Strictly Causal)
    Daily aggregation -> mean -> forward fill ONLY (no interpolation) -> rolling features
    Target: NEXT actual observed creatinine
    """
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
            
            # 1. Consider only observations with timestamp <= t
            history = group.iloc[:i+1]
            
            # 2. Construct the daily representation from that historical prefix
            daily_means = history.groupby('date')['valuenum'].mean()
            
            # 3. Apply forward filling only using information available by t
            if len(daily_means) > 0:
                min_date = daily_means.index.min()
                current_date = t.floor('D')
                full_index = pd.date_range(min_date, current_date, freq='D')
                daily_ffill = daily_means.reindex(full_index).ffill()
                
                # 4. Calculate rolling features only from that causal history
                current_creat = daily_ffill.iloc[-1]
                prev_creat = daily_ffill.iloc[-2] if len(daily_ffill) >= 2 else current_creat
                
                last_3 = daily_ffill.iloc[-3:]
                r_mean = last_3.mean()
                r_std = last_3.std() if len(last_3) >= 2 else 0.0
                if pd.isna(r_std):
                    r_std = 0.0
            else:
                current_creat = np.nan
                prev_creat = np.nan
                r_mean = np.nan
                r_std = 0.0
                
            feature_rows.append({
                'current_creatinine': current_creat,
                'previous_creatinine': prev_creat,
                'creatinine_change': current_creat - prev_creat,
                'rolling_mean': r_mean,
                'rolling_std': r_std
            })
            
        features_df = pd.DataFrame(feature_rows, index=group.index)
        merged = pd.concat([group, features_df], axis=1)
        
        # 5. Map the resulting causal daily features to timestamp t
        valid = merged[merged['is_valid']].copy()
        processed_dfs.append(valid)
        
    if not processed_dfs:
        return pd.DataFrame(), pd.Series()
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    features = ['current_creatinine', 'previous_creatinine', 'creatinine_change', 'rolling_mean', 'rolling_std']
    
    return final_df[['subject_id', 'charttime'] + features], final_df['target']

def preprocess_proposed(df):
    """
    Pipeline B: Proposed
    Exact timestamps -> observation-aware temporal features
    Target: NEXT actual observed creatinine
    """
    df = df.copy()
    processed_dfs = []
    
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('charttime').copy()
        
        group['target'] = group['valuenum'].shift(-1)
        # Unified validation mask: not the first observation, not the last observation
        group['is_valid'] = group['target'].notna() & group['valuenum'].shift(1).notna()
        
        group['current_creatinine'] = group['valuenum']
        group['previous_creatinine'] = group['current_creatinine'].shift(1)
        group['creatinine_change'] = group['current_creatinine'] - group['previous_creatinine']
        group['rolling_mean'] = group['current_creatinine'].rolling(window=3, min_periods=1).mean()
        group['rolling_std'] = group['current_creatinine'].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Temporal features
        group['prev_charttime'] = group['charttime'].shift(1)
        group['delta_t_hours'] = (group['charttime'] - group['prev_charttime']).dt.total_seconds() / 3600.0
        group['delta_t_hours'] = group['delta_t_hours'].fillna(0)
        
        group['time_since_last_observation'] = group['delta_t_hours']
        group['observation_count'] = np.arange(1, len(group) + 1)
        
        total_hours = (group['charttime'] - group['charttime'].iloc[0]).dt.total_seconds() / 3600.0
        group['observation_density'] = group['observation_count'] / (total_hours + 1.0)
        
        group['observation_mask'] = 1.0
        group['observation_confidence'] = np.exp(-0.1 * group['delta_t_hours'])
        
        # Only keep valid rows to match Baseline exactly
        valid = group[group['is_valid']].copy()
        processed_dfs.append(valid)
        
    if not processed_dfs:
        return pd.DataFrame(), pd.Series()
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    features = [
        'current_creatinine', 'previous_creatinine', 'creatinine_change', 
        'rolling_mean', 'rolling_std', 'delta_t_hours', 
        'time_since_last_observation', 'observation_count', 
        'observation_density', 'observation_confidence'
    ]
    
    return final_df[['subject_id', 'charttime'] + features], final_df['target']
