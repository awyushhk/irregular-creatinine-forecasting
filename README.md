# Observation-Aware Irregular-Time Renal Forecasting

An experimental framework for evaluating whether **observation-aware irregular-time preprocessing** improves next-observation creatinine forecasting compared with a **causally constructed daily-aggregation baseline** on the MIMIC-IV Demo dataset.

The experiment focuses on a key challenge in longitudinal clinical data: creatinine measurements are collected at **irregular and patient-dependent time intervals**. Conventional daily aggregation can discard information about the exact timing and availability of observations, while an irregular-time representation can explicitly preserve this temporal information.

## Objective

The primary objective is to compare two temporal preprocessing strategies for predicting the **next actually observed creatinine measurement**:

### Pipeline A — Causal Daily-Aggregation Baseline

A daily-aggregation approach in which temporal information is represented using a daily structure while ensuring that predictive features are constructed **causally**, using only information available at or before the prediction time.

This baseline is an **adapted causal implementation inspired by the preprocessing described in the Eigen-Guided Transformer CKD literature**. It is not intended to be an exact reproduction of the complete preprocessing or forecasting protocol of that work.

### Pipeline B — Observation-Aware Irregular-Time Approach

The proposed approach preserves the **exact timestamps of creatinine observations** instead of forcing measurements into a regular daily grid.

It incorporates observation-aware temporal information such as:

- Time since the previous observation
- Temporal freshness
- Observation availability/information
- Irregular observation intervals

Both pipelines are evaluated on the **same next-observed-creatinine prediction task** and use the same downstream model configuration.

## Experimental Design

The experiment was designed to avoid common sources of temporal leakage and unfair comparison in longitudinal clinical forecasting.

### Prediction Target

For a patient at observation time `t`:

```text
X(t) → Creatinine(t+1)
```

where `t+1` represents the **next actual observed creatinine measurement**, rather than an artificially generated future calendar-day value.

### Causal Temporal Processing

At prediction time `t`, features are constructed using only information satisfying:

```text
observation_timestamp ≤ prediction_timestamp
```

This prevents future measurements from influencing historical features.

In particular, the experiment avoids:

- Future interpolation
- Same-day look-ahead leakage
- Artificial calendar observations being treated as clinical measurements
- Mismatched target timestamps between the two pipelines

### Controlled Comparison

To isolate the effect of temporal representation:

- Both pipelines predict the same target observations.
- Corresponding samples are aligned between pipelines.
- Both use the same `RandomForestRegressor` configuration.
- Missingness is applied to historical observations while keeping the future prediction target fixed.
- Explicit target and leakage checks are performed.

The final validation reported:

```text
Target timestamp mismatches = 0
Target value mismatches     = 0
Future leakage violations   = 0
```

## Dataset

The experiment uses the **MIMIC-IV Demo v2.2** dataset and focuses on blood creatinine measurements.

After filtering and quality checks, the final experiment contained:

```text
98 usable patients
2,999 raw creatinine observations
```

Because the MIMIC-IV Demo dataset is substantially smaller than the full MIMIC-IV database, the results should be interpreted as a **pilot experimental evaluation**, not as evidence of clinical generalizability.

## Architecture

```text
MIMIC-IV Demo
      │
      ▼
Creatinine observations
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
Causal Daily Baseline       Observation-Aware
      │                     Irregular-Time
      │                     Representation
      │                          │
      └──────────┬───────────────┘
                 ▼
        Same Random Forest
                 │
                 ▼
     Next observed creatinine
                 │
                 ▼
       Evaluation & Audit
```

### Project Structure

```text
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluate.py
│
├── dashboard.py
├── README.md
└── requirements / dependencies
```

### Components

**`src/load_data.py`**

Loads, extracts, and filters blood creatinine measurements from MIMIC-IV Demo v2.2.

**`src/preprocessing.py`**

Implements the two temporal preprocessing pipelines:

- Causal daily-aggregation baseline
- Observation-aware irregular-time representation

**`src/model.py`**

Provides the common `RandomForestRegressor` used for both pipelines, allowing the comparison to focus on temporal preprocessing rather than different model architectures.

**`src/evaluate.py`**

Calculates forecasting metrics including:

- MAE
- RMSE
- R²
- MAE reduction/improvement

It also supports comparison across missingness levels.

**`dashboard.py`**

Provides a Streamlit dashboard for exploring:

- Baseline vs. proposed performance
- Missingness stress-test results
- Temporal forecasting metrics
- Comparative performance across sparsity levels

## Missingness Stress Test

To evaluate robustness to increasingly sparse historical observations, the experiment tests six observation-availability conditions:

```text
0%
10%
20%
30%
40%
50%
```

The missingness experiment removes historical observations while keeping the **future prediction target fixed**. This allows both pipelines to be evaluated under the same forecasting task as historical information becomes increasingly sparse.

The purpose is to measure how performance degrades as the observation process becomes less complete.

## Results

The final controlled experiment produced the following **MAE** results:

| Missingness | Causal Baseline MAE | Proposed MAE | MAE Improvement |
| ----------: | ------------------: | -----------: | --------------: |
|          0% |              0.2272 |   **0.2122** |       **6.61%** |
|         10% |              0.2460 |   **0.2328** |       **5.37%** |
|         20% |              0.2732 |   **0.2429** |      **11.09%** |
|         30% |              0.2951 |   **0.2775** |       **5.96%** |
|         40% |              0.3138 |   **0.3075** |       **2.01%** |
|         50% |              0.3387 |   **0.3279** |       **3.19%** |

The observation-aware irregular-time approach achieved **lower MAE at all six tested missingness levels**.

The largest observed MAE improvement was **11.09% at 20% missingness**.

At 40% missingness, the proposed approach still achieved lower MAE, although the corresponding RMSE and R² comparison showed a slight reversal. Therefore, the results should not be interpreted as universal improvement across every metric at every missingness level.

## Interpretation

The results provide pilot evidence that preserving irregular observation timing and observation-aware information can be beneficial for next-observation creatinine forecasting compared with the adapted causal daily-aggregation baseline used in this experiment.

However, several limitations should be considered:

- MIMIC-IV Demo contains a relatively small number of patients and observations.
- The experiment is not a clinical validation study.
- The baseline is an **adapted causal implementation inspired by Eigen-Guided Transformer preprocessing**, not an exact reproduction of the complete original methodology.
- The downstream model is a Random Forest rather than a specialized temporal deep-learning architecture.
- Results may not generalize to the full MIMIC-IV population or other clinical datasets.

Therefore, the main conclusion is:

> **On this MIMIC-IV Demo pilot, an observation-aware irregular-time representation achieved lower MAE than the causally constructed daily-aggregation baseline across the tested 0–50% historical missingness levels.**

These results motivate further evaluation on larger datasets and with dedicated irregular-time forecasting architectures.

## Reproducibility and Leakage Controls

A major part of the experimental design was validating that the comparison remained fair and temporally causal.

The final experiment enforced:

- Same next-observation forecasting target
- Same target timestamps
- Same target values
- Same corresponding samples
- Patient-level splitting
- Identical downstream Random Forest configuration
- No future interpolation
- No same-day look-ahead
- Causal feature construction
- Fixed future targets during missingness testing
- Explicit leakage and target-alignment audits

The final audit reported **zero detected target timestamp mismatches, target value mismatches, and future leakage violations**.

## Installation

Install the required Python dependencies:

```bash
pip install pandas numpy scikit-learn plotly streamlit scipy
```

## Running the Dashboard

From the project root:

```bash
streamlit run dashboard.py
```

The Streamlit dashboard presents the comparative forecasting results and missingness stress-test analysis.

## Research Positioning

This project investigates a specific methodological question:

> **When predicting the next actual renal observation, does preserving irregular observation timing and observation-aware information provide an advantage over a causally constructed daily representation?**

The experiment is designed as a controlled preprocessing comparison rather than a claim that one model architecture is universally superior.

The current results support further investigation of observation-aware temporal representations for irregular clinical time-series forecasting.
