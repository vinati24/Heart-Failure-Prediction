# Trustworthy Heart Disease Prediction: Uncertainty Quantification & Responsible AI Audit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20·%20DiCE%20·%20MAPIE-blueviolet.svg)](#trustworthy-ai-suite)

## Research Context

Clinical risk models are usually reported with one accuracy number. That is not enough in practice: a clinician also needs to know *why* the model made a prediction, *how sure* it is, *whether it works equally well for different patient groups*, and *what would change* the outcome.

This project builds a heart disease risk model and then audits it on all four of these questions.

## Pipeline

```
Data audit ──> Preprocessing ──> Stacking ensemble ──> Conformal calibration ──> Audit suite
                                                                                  ├─ SHAP (why?)
                                                                                  ├─ MAPIE (how sure?)
                                                                                  ├─ Sex-stratified recall (fair?)
                                                                                  └─ DiCE (what would change it?)
```

- **Data audit:** missingness, correlation, category balance, leakage check, random record review
- **Cleaning:** 172 records with `Cholesterol = 0` (not physiologically possible) are treated as missing, not as real values
- **Feature engineering:** heart-rate reserve (`220 − Age − MaxHR`)
- **Preprocessing:** KNN imputation + RobustScaler (numeric), ordinal encoding (ST slope, exercise angina), one-hot encoding (chest pain type, resting ECG, sex)
- **Model:** stacking ensemble of CatBoost, Random Forest and Logistic Regression, with a Logistic Regression meta-learner
- **Split:** 50% train / 25% conformal calibration / 25% held-out test, all stratified

## Results (held-out test set, n = 230)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.936 |
| Accuracy | 0.878 |
| Precision | 0.890 |
| Recall | 0.890 |
| F1 | 0.890 |

Meta-learner weights: CatBoost 2.51, Logistic Regression 1.68, Random Forest 1.59.

## Trustworthy AI Suite

| Check | Method | What it showed |
|-------|--------|----------------|
| Uncertainty | MAPIE split conformal prediction (α = 0.1) | Returns a prediction *set* with a 90% coverage target instead of a single label |
| Fairness | Recall by sex | Male 89.1% vs female 87.5%, within the 5-point threshold. Note: only 193 of 918 patients are female |
| Explanation | SHAP (permutation explainer on the full pipeline) | Global importance and per-feature direction of effect |
| Counterfactuals | DiCE, varying only resting BP, cholesterol and max HR | For test patient 311, DiCE suggested *raising* resting BP by 17 mmHg and max HR by 78 bpm. Mathematically valid, but not clinically sensible. This is why counterfactual output needs clinical constraints and human review before it reaches a patient |
| Calibration | Reliability curve | Included in the notebook |

## Dataset

[Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (fedesoriano, Kaggle): 918 patients and 11 clinical features, combined from five UCI heart disease cohorts. The target is `HeartDisease` (0/1).

## Project Structure

```
Heart-Failure-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── trustworthy_heart_failure_prediction.ipynb   # Full pipeline
└── data/
    └── heart_failure.csv
```

## Getting Started

```bash
git clone https://github.com/vinati24/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction
pip install -r requirements.txt
jupyter notebook notebooks/trustworthy_heart_failure_prediction.ipynb
```

## Limitations

- Single public dataset; no external validation cohort
- Conformal coverage is guaranteed on average, not for each subgroup
- The fairness check covers sex only, and the female subgroup is small

## License

MIT. See [LICENSE](LICENSE).

## Author

**Vinati Nathwani**, MSc AI for Biomedicine and Healthcare, UCL · BTech CSE (Health Informatics), VIT Bhopal
[GitHub](https://github.com/vinati24) · [LinkedIn](https://linkedin.com/in/vinati-nathwani-42b622260)
