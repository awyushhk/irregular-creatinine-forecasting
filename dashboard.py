import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_data import load_and_filter_mimic_data
from src.preprocessing import split_by_patient, apply_missingness
from src.evaluate import run_experiment

st.set_page_config(page_title="RenalTwin-OT", layout="wide")

st.title("RenalTwin-OT: Observation-Aware Temporal Preprocessing for Renal Trajectory Prediction")
st.warning("⚠️ Pilot experiment on MIMIC-IV Demo. Not clinically validated.")

@st.cache_data
def load_data():
    return load_and_filter_mimic_data()

@st.cache_data
def get_main_results(df):
    train_df, val_df, test_df = split_by_patient(df)
    return run_experiment(train_df, val_df, test_df, missing_frac=0.0)

try:
    with st.spinner("Loading MIMIC-IV Demo Data..."):
        df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["DATA QUALITY", "PATIENT TRAJECTORY", "MODEL COMPARISON", "ROBUSTNESS"])

# --- TAB 1: DATA QUALITY ---
with tab1:
    st.header("Data Quality")
    num_patients = df['subject_id'].nunique()
    raw_obs = len(df)
    
    daily_obs_df = df.groupby(['subject_id', df['charttime'].dt.floor('D')])
    daily_obs = len(daily_obs_df)
    
    compression_pct = (raw_obs - daily_obs) / raw_obs * 100
    
    # Calculate intervals
    intervals = []
    for _, group in df.groupby('subject_id'):
        sorted_times = group['charttime'].sort_values()
        diffs = sorted_times.diff().dt.total_seconds().dropna() / 3600.0
        intervals.extend(diffs.tolist())
    
    intervals = np.array(intervals)
    median_interval = np.median(intervals)
    mean_interval = np.mean(intervals)
    max_interval = np.max(intervals)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Patients", f"{num_patients}")
    col1.metric("Raw Creatinine Observations", f"{raw_obs}")
    col1.metric("Daily Observations", f"{daily_obs}")
    
    col2.metric("Temporal Compression", f"{compression_pct:.1f} %")
    col2.metric("Median Observation Interval", f"{median_interval:.2f} hrs")
    
    col3.metric("Mean Observation Interval", f"{mean_interval:.2f} hrs")
    col3.metric("Maximum Observation Interval", f"{max_interval:.2f} hrs")

# --- TAB 2: PATIENT TRAJECTORY ---
with tab2:
    st.header("Patient Trajectory")
    patients = df['subject_id'].unique()
    selected_patient = st.selectbox("Select Patient (subject_id)", patients)
    
    patient_df = df[df['subject_id'] == selected_patient].copy()
    patient_df = patient_df.sort_values('charttime')
    
    # Compute daily aggregated trajectory
    patient_df['date'] = patient_df['charttime'].dt.floor('D')
    daily_df = patient_df.groupby('date')['valuenum'].mean().reset_index()
    
    if len(daily_df) > 0:
        min_date = daily_df['date'].min()
        max_date = daily_df['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_full = daily_df.set_index('date').reindex(all_dates)
        daily_full['valuenum'] = daily_full['valuenum'].ffill().interpolate(method='linear')
        daily_full = daily_full.reset_index().rename(columns={'index': 'date'})
    else:
        daily_full = pd.DataFrame(columns=['date', 'valuenum'])
    
    patient_df['prev_time'] = patient_df['charttime'].shift(1)
    patient_df['time_gap_hrs'] = (patient_df['charttime'] - patient_df['prev_time']).dt.total_seconds() / 3600.0
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Original timestamped creatinine measurements (actual observations as markers)
    fig.add_trace(go.Scatter(x=patient_df['charttime'], y=patient_df['valuenum'], mode='markers', name='Actual Observations', marker=dict(size=10, color='red')), row=1, col=1)
    
    # Daily aggregated trajectory
    if not daily_full.empty:
        fig.add_trace(go.Scatter(x=daily_full['date'], y=daily_full['valuenum'], mode='lines', name='Daily Aggregated Trajectory', line=dict(color='blue', dash='dash')), row=1, col=1)
    
    # Time gaps between observations
    fig.add_trace(go.Bar(x=patient_df['charttime'], y=patient_df['time_gap_hrs'], name='Time Gaps (Hours)', marker_color='orange'), row=2, col=1)
    
    fig.update_layout(height=600, title=f"Trajectory for Patient {selected_patient}", hovermode='x unified')
    fig.update_yaxes(title_text="Creatinine (mg/dL)", row=1, col=1)
    fig.update_yaxes(title_text="Gap (hrs)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: MODEL COMPARISON ---
with tab3:
    st.header("Model Comparison")
    
    with st.spinner("Training models and computing metrics..."):
        results = get_main_results(df)
        
    baseline = results['Baseline']
    proposed = results['Proposed']
    
    metrics = ['MAE', 'RMSE', 'R2']
    data = []
    for m in metrics:
        b_val = baseline[m]
        p_val = proposed[m]
        change = p_val - b_val
        change_str = f"{change:+.4f}"
        data.append([m, f"{b_val:.4f}", f"{p_val:.4f}", change_str])
        
    res_df = pd.DataFrame(data, columns=['Metric', 'Baseline', 'Proposed', 'Change'])
    
    st.table(res_df)
    
    mae_reduction = results['MAE_Reduction_Pct']
    st.metric("Percentage MAE Reduction", f"{mae_reduction:.2f}%")

# --- TAB 4: ROBUSTNESS ---
with tab4:
    st.header("Robustness")
    
    if st.button("Run Robustness Test"):
        missing_rates = [10, 20, 30, 40, 50]
        b_maes = []
        p_maes = []
        
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, rate in enumerate(missing_rates):
            status_text.text(f"Testing {rate}% missingness...")
            train_df, val_df, test_df = split_by_patient(df)
            
            frac = rate / 100.0
            train_df_m = apply_missingness(train_df, frac)
            val_df_m = apply_missingness(val_df, frac)
            test_df_m = apply_missingness(test_df, frac)
            
            res = run_experiment(train_df_m, val_df_m, test_df_m)
            b_maes.append(res['Baseline']['MAE'])
            p_maes.append(res['Proposed']['MAE'])
            
            progress.progress((i + 1) / len(missing_rates))
            
        status_text.text("Robustness test complete.")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=missing_rates, y=b_maes, mode='lines+markers', name='Baseline (Daily Agg + Impute)'))
        fig2.add_trace(go.Scatter(x=missing_rates, y=p_maes, mode='lines+markers', name='Proposed (Observation-Aware)'))
        fig2.update_layout(
            title="Impact of Missingness on Model Performance (MAE)",
            xaxis_title="Missingness %",
            yaxis_title="MAE",
            xaxis=dict(tickvals=missing_rates)
        )
        st.plotly_chart(fig2, use_container_width=True)
