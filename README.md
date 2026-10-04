# Irregular-Time Causal Renal Trajectory Forecasting

> **Experimental research prototype.** This repository evaluates whether preserving irregular observation timing and causal renal trajectory information improves next-observation creatinine forecasting compared with daily-aggregation baselines on MIMIC-IV Demo. It forms a methodological foundation for a future CKD Digital Twin / Module 1 system and is **not** the complete Module 1 implementation.

---

## Research Question

> *Can preserving irregular observation timing and causal renal trajectory information improve prediction of the next actually observed creatinine measurement compared with causally constructed daily representations?*

The contribution is a controlled renal-specific experimental framework that audits causal preprocessing pipelines, isolates the value of irregular-time representation, and evaluates robustness under synthetic missingness — not a claim that irregular-time preprocessing is a novel invention.

---

## Repository Overview

```
irregular-creatinine-forecasting/
├── src/
│   ├── load_data.py       # MIMIC-IV Demo data extraction and filtering
│   ├── preprocessing.py   # Three causal preprocessing pipelines (A, B, C)
│   ├── model.py           # RandomForestRegressor wrapper
│   └── evaluate.py        # Experiment runner with runtime alignment assertions
├── dashboard.py           # Streamlit dashboard (5 tabs, frozen results)
├── README.md
└── data/
    └── mimic-iv-clinical-database-demo-2.2/
        └── hosp/
            ├── labevents.csv.gz
            └── d_labitems.csv.gz
```

---

## Dataset

**Source:** MIMIC-IV Clinical Database Demo v2.2 (PhysioNet)  
**Lab item:** Creatinine (serum) — item ID verified programmatically from `d_labitems.csv.gz`; not assumed.  
**Inclusion criterion:** Patients with at least 3 creatinine observations.  
**Final cohort:** 98 usable patients, 2,999 raw creatinine observations.

> ⚠️ **Pilot dataset.** MIMIC-IV Demo contains 100 patients. Results from this pilot study should not be interpreted as clinical validation or as representative of population-level creatinine dynamics.

---

## Forecasting Target

```
X(t)  →  Creatinine(t+1)
```

where **t+1 is the next actual observed creatinine measurement** — not a synthetic next-calendar-day interpolation. The target is constructed via `shift(-1)` on the sorted observation sequence per patient, strictly preserving the clinical irregularity of observation timing.

---

## Experimental Design

### Patient Split

Patient-level splitting with no row-level leakage across sets:

| Set | Fraction |
|---|---|
| Train | 70% of patients |
| Validation | 15% of patients |
| Test | 15% of patients |

### Multi-Seed Evaluation

Five fixed random seeds: **42, 123, 456, 789, 2026**  
Each seed produces an independent patient-level split. Results are aggregated as **mean ± standard deviation** across the five seeds.

### Missingness Stress Test

Six synthetic missingness levels: **0%, 10%, 20%, 30%, 40%, 50%**

Interior observations are randomly removed at the specified rate. The first and last observations per patient are preserved. This is a **synthetic random-masking stress test** and does not reproduce real clinical missingness mechanisms such as informative observation, scheduled testing, or disease-driven measurement frequency.

### Downstream Model (Identical for All Pipelines)

```python
RandomForestRegressor(n_estimators=100, random_state=42)
```

All three pipelines use the same model configuration, hyperparameters, and evaluation procedure. The experiment compares preprocessing representations, not model architectures.

---

## Three Preprocessing Pipelines

### Pipeline A — Paper-Inspired Daily Baseline

Adapted from the broad preprocessing philosophy described in daily-aggregation clinical ML literature, particularly the organisation of creatinine measurements into daily bins as described in Saeed & Aldera (2026) (see [Literature](#literature)). **This is not an exact reproduction** of that paper's full preprocessing protocol, model architecture, or forecasting setup — our prediction target, patient cohort, and downstream model differ substantially.

**Steps (all steps strictly causal — no future information used at any step):**

1. Floor observation timestamps to calendar day
2. Compute daily mean creatinine from observations **at or before** prediction time `t`
3. Build a daily grid from patient start to `day(t)`
4. **Constrained interpolation:** gaps of **≤ 7 missing days** are filled by linear interpolation between the two known endpoints; gaps **> 7 days** are not interpolated and are instead forward-filled
5. Compute rolling features on the causal daily grid
6. Map causal daily features back to the exact observation timestamp `t`

**Features:** `current_creatinine`, `previous_creatinine`, `creatinine_change`, `rolling_mean`, `rolling_std`

**Leakage controls:**
- Daily grid is rebuilt at each prediction timestamp using only historical prefix `group.iloc[:i+1]`
- Audited: modifying a future observation does not alter any feature at a past timestamp
- Same-day future observations cannot enter the daily grid

---

### Pipeline B — Strict Causal Daily Baseline

The most conservative daily-aggregation baseline — no interpolation at any gap length.

**Steps (strictly causal):**

1. Floor timestamps to calendar day
2. Daily mean from observations **at or before** `t`
3. Causal forward-fill of all gaps — no interpolation regardless of gap duration
4. Rolling features on causal daily grid
5. Map back to exact observation timestamp `t`

**Features:** `current_creatinine`, `previous_creatinine`, `creatinine_change`, `rolling_mean`, `rolling_std`

---

### Pipeline C — Irregular-Time Causal Renal Trajectory Representation

Preserves exact irregular observation timestamps. Does not collapse observations into a daily grid. All features are constructed from current and past observations only.

**Final feature set (ablation-justified — see [Ablation Study](#ablation-study)):**

| Feature | Description |
|---|---|
| `current_creatinine` | Creatinine value at time `t` |
| `previous_creatinine` | Creatinine value at the immediately preceding observation |
| `creatinine_change` | `current_creatinine` − `previous_creatinine` |
| `delta_t_hours` | Hours elapsed since the previous observation (from `shift(1)` on `charttime`) |
| `rolling_mean` | 3-observation causal rolling mean over the patient's history up to `t` |
| `rolling_std` | 3-observation causal rolling standard deviation up to `t` |

**Features evaluated and removed during ablation:**

| Feature | Reason for removal |
|---|---|
| `observation_mask` | Confirmed constant = 1.0 for all evaluated rows (only actual observations are included); zero discriminative power |
| `observation_count` | Did not improve performance under the evaluated protocol; conflates trajectory position with elapsed time |
| `observation_density` | Most consistently harmful addition in ablation; collinear with `observation_count` and elapsed time |
| `time_since_last_observation` | Duplicate of `delta_t_hours` |
| `observation_confidence` | Monotone transformation of `delta_t_hours`; adds no orthogonal information |

**Leakage controls:**
- `previous_creatinine` and `creatinine_change` use `shift(1)` — strictly past
- `delta_t_hours` computed from `shift(1)` charttime — strictly causal
- `rolling_mean` and `rolling_std` computed on sorted history with `min_periods ≥ 1` — no future values

---

## Alignment Assertions (Enforced at Runtime)

All three pipelines are required to evaluate on the **same test observations** at every run:

```python
assert len(y_test_a) == len(y_test_b) == len(y_test_c)
assert (X_test_a['charttime'].values == X_test_b['charttime'].values).all()
assert (X_test_b['charttime'].values == X_test_c['charttime'].values).all()
assert (X_test_a['subject_id'].values == X_test_b['subject_id'].values).all()
assert np.max(np.abs(y_test_a.values - y_test_b.values)) < 1e-6  # A/B target match
assert np.max(np.abs(y_test_b.values - y_test_c.values)) < 1e-6  # B/C target match
```

---

## Final Results (5-Seed Multi-Seed Evaluation)

Results shown as **mean ± std** across five independent random seeds. Approximately 315 evaluation samples at 0% missingness, decreasing with higher missingness rates. Sample counts are evaluation samples, not patient counts.

### 0% Missingness (~315 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A — Paper-Inspired Daily | 0.2644 ± 0.1249 | 0.4473 ± 0.2592 | 0.8874 ± 0.0345 |
| B — Strict Causal Daily | 0.2611 ± 0.1240 | 0.4407 ± 0.2598 | 0.8912 ± 0.0352 |
| **C — Irregular-Time Causal** | **0.2498 ± 0.1220** | **0.4231 ± 0.2591** | **0.9003 ± 0.0344** |

### 10% Missingness (~288 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A | 0.2763 ± 0.1276 | 0.4593 ± 0.2577 | 0.8796 ± 0.0335 |
| B | 0.2741 ± 0.1242 | 0.4563 ± 0.2614 | 0.8818 ± 0.0349 |
| **C** | **0.2582 ± 0.1260** | **0.4373 ± 0.2694** | **0.8923 ± 0.0416** |

### 20% Missingness (~252 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A | 0.2939 ± 0.1242 | 0.4936 ± 0.2652 | 0.8603 ± 0.0283 |
| B | 0.2889 ± 0.1257 | 0.4824 ± 0.2700 | 0.8679 ± 0.0364 |
| **C** | **0.2721 ± 0.1261** | **0.4486 ± 0.2647** | **0.8861 ± 0.0348** |

### 30% Missingness (~218 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A | 0.3105 ± 0.1355 | 0.5200 ± 0.2830 | 0.8455 ± 0.0369 |
| B | 0.3051 ± 0.1391 | 0.5068 ± 0.2984 | 0.8576 ± 0.0408 |
| **C** | **0.2973 ± 0.1450** | **0.4868 ± 0.3020** | **0.8699 ± 0.0470** |

### 40% Missingness (~183 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A | 0.3268 ± 0.1256 | 0.5366 ± 0.2598 | 0.8267 ± 0.0418 |
| B | 0.3186 ± 0.1330 | 0.5215 ± 0.2727 | 0.8401 ± 0.0370 |
| **C** | **0.3019 ± 0.1349** | **0.5064 ± 0.2767** | **0.8503 ± 0.0425** |

### 50% Missingness (~147 samples)

| Pipeline | MAE | RMSE | R² |
|---|---|---|---|
| A | 0.3563 ± 0.1501 | 0.5962 ± 0.3067 | 0.7982 ± 0.0427 |
| B | 0.3629 ± 0.1584 | 0.5845 ± 0.3171 | 0.8098 ± 0.0356 |
| **C** | **0.3481 ± 0.1641** | **0.5692 ± 0.3222** | **0.8223 ± 0.0364** |

---

## Pipeline C Win Rate

Win rate measures how often Pipeline C achieves a lower MAE than each baseline across the five seeds. This is a descriptive comparison across random splits and is **not** a formal statistical significance test.

| Missingness | C beats A | C beats B |
|---|---|---|
| 0% | 5/5 | 4/5 |
| 10% | 5/5 | 5/5 |
| 20% | 5/5 | 4/5 |
| 30% | 4/5 | 3/5 |
| 40% | 5/5 | 5/5 |
| 50% | 2/5 | 5/5 |

Under the evaluated protocol, Pipeline C achieves a lower MAE than Pipeline A in 26/30 seed × missingness combinations and lower than Pipeline B in 26/30 combinations. Results at 30% and 50% missingness show greater variability across seeds.

---

## Ablation Study

A controlled progressive ablation was conducted on Pipeline C to determine which features contribute to predictive performance. Features were added one at a time in a strict hierarchy. All ablation configurations used the same seeds, patient splits, missingness levels, Random Forest, and forecasting targets. **No test-set tuning was performed.**

| Config | Feature Set | Note |
|---|---|---|
| A0 | `current_creatinine` | Baseline |
| A1 | + `previous_creatinine` | Generally increases MAE slightly; more useful under high missingness |
| A2 | + `creatinine_change` | Reduced MAE relative to A1 at every missingness level |
| A3 | + `delta_t_hours` | Mixed benefit; more helpful at higher missingness (≥20%) |
| A4 | + `observation_mask` | Confirmed constant = 1.0; zero discriminative power; delta ≤ 0.0002 MAE |
| A5 | + `observation_count` | Generally degraded performance |
| A6 | + `observation_density` | Most consistently harmful addition; increased MAE at every missingness level |
| A7 | + `rolling_mean` | Recovered performance relative to A6 |
| A8 | + `rolling_std` | Further improved performance; this configuration is the final Pipeline C |

**Key observations from ablation:**

- `observation_mask` was confirmed constant at 1.0 across all evaluated rows and therefore carries zero discriminative power in this setup.
- `creatinine_change` improved MAE relative to the previous configuration at every missingness level tested.
- `delta_t_hours` was not uniformly beneficial; it tended to provide more improvement under higher missingness (≥20%).
- `observation_count` and `observation_density` generally degraded performance and were excluded from the final representation.
- Rolling statistics (`rolling_mean`, `rolling_std`) consistently recovered performance after the degradation introduced by observation-awareness features alone.

The final reduced Pipeline C (A8 without mask/count/density) improved mean MAE relative to the original extended Pipeline C at every missingness level.

---

## Leakage Audit

The following checks were applied and passed during the final experiment run:

| Check | Result |
|---|---|
| Patient overlap between train / val / test | 0 overlapping patients |
| Feature timestamp after prediction timestamp | 0 violations |
| Same-day future observations in baseline features | 0 violations |
| Interpolation leakage in Pipeline A | 0 violations |
| Gaps > 7 days interpolated in Pipeline A | 0 violations |
| Target observation used in feature construction | 0 violations |
| Target value mismatch across A / B / C | max difference < 1e-6 |
| Prediction timestamp mismatch across pipelines | 0 mismatches |
| Subject ID mismatch across pipelines | 0 mismatches |
| Test-set tuning of any preprocessing parameter | None performed |

> **No detected leakage under the implemented audit checks.** Causal correctness was enforced through code structure (iterative historical prefix construction at each prediction timestamp) and verified through automated modification tests (changing a future observation does not alter any past feature value in Pipeline A or C).

---

## Literature

### Primary Inspiration for Pipeline A

Saeed, M. S., & Aldera, H. (2026). *Eigen-guided transformer: A data-driven approach for chronic kidney disease forecasting.* PLOS ONE. https://doi.org/10.1371/journal.pone.0322569

This work organises EHR creatinine measurements into daily bins for trajectory modelling and identifies daily aggregation as a limitation that can obscure intra-day dynamics. Pipeline A is an **adapted causal implementation** inspired by the broad preprocessing philosophy described therein. It is not a reproduction of the paper's model architecture, training procedure, or evaluation protocol.

### Broader Context

This project is informed by the broader literature on observation-aware and irregular-time EHR modelling (e.g., GRU-D, SeFT, mTAN). The use of `delta_t_hours` and observation-mask concepts is drawn from that literature. This repository does not claim to have invented these representations; rather, it evaluates their utility in a controlled renal-specific context with explicit causal auditing.

---

## Streamlit Dashboard

The project includes a self-contained Streamlit dashboard (`dashboard.py`) with five tabs:

| Tab | Contents |
|---|---|
| **Overview** | Dataset description, patient split, model configuration, caveats |
| **Methodology** | Side-by-side description of Pipelines A, B, C with leakage controls and feature lists |
| **Performance** | MAE / RMSE / R² plots with ±1 std shaded bands, full results table, win-rate table |
| **Ablation** | Progressive ablation table, incremental MAE bar chart, justification for feature removal decisions |
| **Audit** | All leakage check results, runtime assertion code, limitations table, experiment provenance |

Final frozen results are embedded directly in the dashboard. The dashboard does not re-run the full experiment on load and is therefore fast and self-contained.

---

## Installation and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `plotly`, `streamlit`

### Run the Dashboard

```bash
streamlit run dashboard.py
```

### Re-run the Experiment

```python
# Example: single seed, single missingness level
from src.load_data import load_and_filter_mimic_data
from src.preprocessing import split_by_patient, apply_missingness
from src.evaluate import run_experiment

df = load_and_filter_mimic_data('data/mimic-iv-clinical-database-demo-2.2')
train_df, val_df, test_df = split_by_patient(df, seed=42)
results = run_experiment(train_df, val_df, test_df, missing_frac=0.0)
```

---

## Limitations

The following limitations apply to all results and conclusions drawn from this repository:

1. **Dataset size.** MIMIC-IV Demo contains 100 patients. This is a pilot study; results may not generalise to larger or different patient populations, EHR systems, or clinical contexts.

2. **Synthetic missingness.** The missingness experiment uses uniform random removal of interior observations. Real clinical missingness is informative and structured — it reflects test ordering patterns, disease severity, hospital workflows, and other factors not captured by random masking.

3. **Single biomarker.** Only serum creatinine is modelled. Other clinically relevant CKD biomarkers (eGFR, BUN, cystatin C, electrolytes, proteinuria) are not included in this pilot.

4. **Downstream model.** Random Forest is used as a common, interpretable downstream model for representation comparison. Results may differ substantially with temporal deep learning architectures (LSTM, Transformer, GRU-D) that can natively exploit irregular time series structure.

5. **Prediction horizon.** Only the next single observation is predicted. Multi-step trajectory forecasting and longer-horizon CKD progression modelling are not evaluated here.

6. **No external validation.** All evaluation is performed on held-out test patients from the same MIMIC-IV Demo dataset. No external cohort, prospective validation, or independent replication has been conducted.

7. **No clinical validation.** The irregular-time causal representation has not been validated as a latent physiological or nephron state. No clinical decisions should be made based on these experimental findings.

8. **No formal statistical testing.** Results are reported as mean ± standard deviation across five random seeds. No confidence intervals, hypothesis tests, or corrections for multiple comparisons have been applied.

9. **Scope.** This is a preprocessing and representation experiment, not the complete CKD Digital Twin. Biomarker integration, state estimation, clinical decision support, and longitudinal trajectory modelling are not implemented here.

---

## Research Position

This repository is the experimental foundation for a larger CKD Digital Twin / Module 1 research project. Its purpose is to establish and audit a causal irregular-time renal trajectory representation before integrating additional biomarkers, state estimation, and downstream CKD/Digital Twin components.

The results provide evidence — under this specific pilot protocol — that preserving irregular observation timing and causal trajectory history supports improved next-observation creatinine forecasting compared with daily-aggregation baselines. This supports further investigation in larger cohorts and with more sophisticated temporal models.

---

*MIMIC-IV Demo data accessed via PhysioNet under the PhysioNet Credentialed Health Data License.*
