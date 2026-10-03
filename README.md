# RenalTwin-OT: Observation-Aware Temporal Preprocessing for Renal Trajectory Prediction

This is a 3-hour research prototype MVP designed to evaluate the impact of observation-aware temporal preprocessing (OT) versus traditional daily aggregation and imputation on predicting renal trajectories.

## Objective
To conduct a controlled experiment comparing two preprocessing pipelines for predicting the next observed creatinine value:
1. **Pipeline A (Baseline)**: Daily aggregation, forward filling, and linear interpolation (approximating common approaches like those in the 2026 Eigen-Guided Transformer CKD paper).
2. **Pipeline B (Proposed)**: Utilizing exact timestamps without binning, incorporating observation-aware temporal features (e.g., temporal freshness / observation confidence).

## Architecture
- **src/load_data.py**: Extracts and filters blood creatinine from the MIMIC-IV Demo v2.2 dataset.
- **src/preprocessing.py**: Implements the Baseline (Pipeline A) and Proposed (Pipeline B) temporal logic. 
- **src/model.py**: Uses `RandomForestRegressor` for fair comparison using identical hyperparameters.
- **src/evaluate.py**: Computes MAE, RMSE, R², and calculates the MAE reduction percentage.
- **dashboard.py**: A Streamlit dashboard presenting the temporal metrics and a Missingness Stress Test.

## How to Run
Ensure you have the required dependencies:
```bash
pip install pandas numpy scikit-learn plotly streamlit scipy
```

Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

## Missingness Stress Test
The dashboard includes an experiment where 10%, 20%, 30%, 40%, and 50% of the observations are artificially hidden. We compare the robustness of the Baseline versus the Proposed model under increasing sparsity.

*Note: The MIMIC-IV Demo dataset is very small (~100 patients with >=3 measurements). Model performance metrics are primarily for demonstrating the pipeline and may not reflect clinically robust results.*
