# Diabetes Hospital Readmission Prediction (30-Day)

Predict which diabetic patients are likely to be readmitted to hospital within 30 days of discharge, so care teams could prioritize follow-up for high-risk patients.

## Problem

Hospital readmissions are costly and often preventable. For diabetic patients, a return within 30 days is a standard quality metric. This project builds a baseline machine-learning model for 30-day readmission risk and examines which factors are associated with it.

## Dataset

- **Source:** UCI Machine Learning Repository — *Diabetes 130-US Hospitals for Years 1999-2008* (101,766 encounters from 130 US hospitals).
- **Target:** `readmitted == "<30"` → 1 (readmitted within 30 days), otherwise 0 (`">30"` or `"NO"`).
- **Class balance:** only **11.2%** of encounters are positive, so accuracy alone is misleading — this project reports precision, recall, F1, and ROC-AUC.
- **Features used (interpretable subset):**
  - Numeric: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`
  - Categorical: `age`, `gender`, admission/discharge categories, `A1Cresult`, `insulin`, `change`, `diabetesMed`, `metformin`, `glyburide`, `glipizide`
- Missing values are encoded as `"?"` in the raw file and are handled with median (numeric) / most-frequent (categorical) imputation inside the pipeline.

## Method

1. Load data, replace `"?"` with missing, create the binary target.
2. Exploratory analysis (full dataset): readmission rate by age group and by prior inpatient visits.
3. To keep run time manageable, train on a **stratified random sample of 30,000 encounters** (80/20 train/test split, stratified). The full 101,766 rows can be used by setting `SAMPLE_N = None` in the script.
4. Preprocessing with a scikit-learn `ColumnTransformer` (scaling + one-hot encoding).
5. Two models compared, both using balanced class weights: **Logistic Regression** and **Random Forest** (300 trees).

## Results (real, from `outputs/metrics.txt`)

Modeling sample: 30,000 encounters — train 24,000, test 6,000. Positive rate: 11.2%.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.684 | 0.179 | 0.512 | 0.266 | **0.661** |
| Random Forest | 0.888 | 0.333 | 0.002 | 0.003 | 0.626 |

**How to read this honestly:**
- Random Forest's 88.8% accuracy is misleading: it predicts almost everyone as "not readmitted" (recall ≈ 0), so it catches almost no real readmissions.
- Logistic Regression catches about **51%** of true 30-day readmissions (recall 0.512), at the cost of many false alarms (precision 0.179). For a screening/follow-up tool, recall usually matters more — missing a high-risk patient is worse than an extra follow-up call.
- ROC-AUC ≈ 0.66 means the model ranks a random readmitted patient above a random non-readmitted one about 66% of the time: a useful baseline, not a deployable clinical tool.

**What the data shows (full dataset):**
- Prior inpatient visits are the clearest signal: readmission rises from ~8.5% (0 prior visits) to ~13% (1), ~17% (2), and ~26% (3+).
- Readmission rate is fairly flat (~10–12%) across adult age groups, slightly higher in the 20–30 group (~14%), and much lower in children.
- Random Forest feature importance ranks `num_lab_procedures`, `num_medications`, `time_in_hospital`, `number_inpatient`, `num_procedures`, and `number_diagnoses` highest — all proxies for how sick/complex the patient was during the stay.

## Charts (in `outputs/`)

- `readmission_rate_by_age.png` — readmission rate by age group
- `readmission_rate_by_prior_inpatient.png` — readmission rate by prior inpatient visits
- `feature_importance.png` — top 15 Random Forest features
- `confusion_matrix.png` — confusion matrix for the best model (Logistic Regression)

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python predict_readmission.py
```

Outputs are written to `outputs/` (`metrics.json`, `metrics.txt`, charts).

## Skills demonstrated

Python, pandas, scikit-learn (pipelines, imputation, one-hot encoding, logistic regression, random forest), matplotlib, class-imbalance-aware evaluation, reproducible sampling.

## Clinical interpretation

The strongest, most clinically sensible signal is **prior utilization**: patients already cycling through hospital stays are far more likely to return within 30 days. Medication count and lab-procedure count act as severity markers. A model like this would be used to *triage follow-up resources* (earlier clinic visits, medication review, glucose monitoring support), never to make treatment decisions on its own.

## Fairness & limitations

- The dataset is from **1999–2008** US hospitals; practice patterns, medications, and coding have changed since.
- Race and other sensitive attributes were deliberately **not** used as features, but correlated features (e.g., admission source, discharge disposition) can still act as proxies — a fairness audit by subgroup would be required before any real use.
- The data are encounters, not unique patients; repeated patients can appear in both train and test, which can inflate performance.
- No external validation, no calibration analysis, and the 0.5 decision threshold was not tuned. ROC-AUC of ~0.66 is a baseline result: the value of this project is the honest pipeline and evaluation, not a claim of clinical readiness.

## Author

Faezeh Ahmadzadeh — Data Science | Applied ML | Insurance, Healthcare & Biotech
