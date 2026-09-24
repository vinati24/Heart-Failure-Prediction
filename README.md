# Trustworthy Heart Failure Prediction: Uncertainty Quantification & Responsible AI Audit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20·%20DiCE%20·%20MAPIE-blueviolet.svg)](#explainability--trustworthy-ai-suite)

## Research Context

Standard machine learning pipelines for clinical risk prediction typically report a single accuracy metric and treat the model as a black box. In high-stakes medical settings, this is insufficient — clinicians need to understand *why* a prediction was made, *what would need to change* for a different outcome, and *how confident* the model actually is.

This project develops a **rigorously calibrated** heart failure prediction pipeline and subjects it to a **systematic responsible-AI evaluation**, demonstrating that high aggregate accuracy alone is insufficient for trustworthy clinical deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │ KNN Imputer  │→ │ RobustScaler │→ │ ColumnTransformer │    │
│  │ (missing     │  │ (outlier-    │  │ (categorical +    │    │
│  │  values)     │  │  resistant)  │  │  numerical)       │    │
│  └──────────────┘  └──────────────┘  └───────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    STACKED ENSEMBLE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │ Random Forest│  │   CatBoost   │  │      SVM         │     │
│  │  (Base 1)    │  │  (Base 2)    │  │   (Base 3)       │     │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘     │
│         └──────────────────┼───────────────────┘               │
│                    ┌───────▼───────┐                           │
│                    │   Logistic    │                           │
│                    │  Regression   │                           │
│                    │ (Meta-Learner)│                           │
│                    └───────┬───────┘                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              TRUSTWORTHY AI EVALUATION SUITE                   │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │     SHAP     │  │     DiCE     │  │      MAPIE       │     │
│  │  (Why this   │  │ (What would  │  │  (How confident  │     │
│  │  prediction?)│  │  change it?) │  │  is the model?)  │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | 0.936 | Stacked Ensemble (RF + CatBoost + LR) |
| **Accuracy** | 87.5% | 5-fold stratified cross-validation |
| **Precision** | 0.89 | Weighted average across classes |
| **Recall** | 0.88 | Weighted average across classes |

### Explainability & Trustworthy AI Suite

| Method | Purpose | Key Finding |
|--------|---------|-------------|
| **SHAP** (Global) | Feature importance ranking | Ejection fraction, serum creatinine, and time dominate predictions — consistent with clinical cardiology literature |
| **SHAP** (Local) | Patient-level explanation | Individual force plots reveal per-patient risk drivers for clinical communication |
| **DiCE** (Counterfactuals) | Actionable "what-if" scenarios | ⚠️ Exposed a **critical failure mode**: model generated biologically impossible interventions (e.g., reducing age by 20 years), motivating human-in-the-loop oversight |
| **MAPIE** (Conformal Prediction) | Calibrated uncertainty intervals | Replaces point predictions with distribution-free, coverage-guaranteed intervals at 95% confidence |

## Project Structure

```
Heart-Failure-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── trustworthy_heart_failure_prediction.ipynb   # Full pipeline
└── data/
    └── heart_failure.csv                            # UCI Heart Failure Dataset
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/vinati24/Heart-Failure-Prediction.git
cd Heart-Failure-Prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/trustworthy_heart_failure_prediction.ipynb
```

## Methodology

### 1. Exploratory Data Analysis
- Missing value analysis using `missingno` correlation matrices
- Statistical distribution checks and outlier identification
- Feature correlation analysis

### 2. Preprocessing Pipeline
- `ColumnTransformer` for automated scaling (RobustScaler) and encoding
- KNN Imputation for clinically-informed missing value handling
- Stratified train/test split preserving class distribution

### 3. Model Development
- Individual base learners: Random Forest, CatBoost, SVM
- `StackingClassifier` with Logistic Regression meta-learner
- Hyperparameter tuning via cross-validation

### 4. Responsible AI Evaluation
- **SHAP**: TreeExplainer for global feature importance (summary plots) and local explanations (force plots, waterfall plots)
- **DiCE**: Diverse counterfactual generation revealing actionable vs. impossible interventions
- **MAPIE**: Conformal prediction intervals providing mathematically rigorous uncertainty bounds

## Dataset

The dataset is derived from the [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) dataset, containing 299 patient records with 13 clinical features including ejection fraction, serum creatinine, age, anaemia status, and follow-up time.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nathwani2024trustworthy,
  author = {Nathwani, Vinati},
  title = {Trustworthy Heart Failure Prediction: Uncertainty Quantification and Responsible AI Audit},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/vinati24/Heart-Failure-Prediction}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Vinati Nathwani**
- MSc AI for Biomedicine and Healthcare — University College London (UCL)
- BTech Computer Science (Health Informatics) — VIT Bhopal
- [GitHub](https://github.com/vinati24) · [LinkedIn](https://linkedin.com/in/vinati-nathwani-42b622260)
