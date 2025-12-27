# Heart Disease Prediction & Explainable AI (XAI) Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20%26%20DiCE-green.svg)](https://github.com/marcotcr/lime)

## Project Overview
This repository contains a machine learning pipeline for predicting heart disease. Unlike standard classification projects, this implementation prioritizes trustworthiness by integrating model interpretability and uncertainty quantification. 

The goal is to provide healthcare professionals not just with a prediction, but with a clear "why" and a measure of "how confident" the model is.

## Key Features
- **Stacked Ensemble Architecture**: Utilizes a `StackingClassifier` combining Random Forest, CatBoost, and Logistic Regression for superior predictive performance.
- **Model Interpretability (XAI)**:
  - **SHAP (SHapley Additive exPlanations)**: For global and local feature importance analysis.
  - **DiCE (Diverse Counterfactual Explanations)**: Generates "what-if" scenarios to identify what changes in a patient's profile would lead to a different diagnosis.
- **Uncertainty Quantification**: Powered by **MAPIE** (Conformal Prediction) to provide rigorous prediction intervals.
- **Advanced Preprocessing**: Robust scaling, KNN Imputation for missing data, and specialized handling of categorical variables.

## Tech Stack
- **Core ML**: `scikit-learn`, `CatBoost`
- **Explainability**: `shap`, `dice-ml`
- **Uncertainty**: `mapie`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`, `missingno`

## Workflow
1. **EDA**: Missing value analysis and statistical distribution checks.
2. **Preprocessing**: Pipeline integration using `ColumnTransformer` for automated scaling and encoding.
3. **Training**: Development of base learners and meta-learner (Stacking).
4. **Validation**: Evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. **Interpretability**: Generating SHAP summary plots and counterfactual instances for specific patient cases.

## 📈 Key Results
The project demonstrates that while ensemble models provide superior accuracy, the integration of SHAP and DiCE provides the necessary clinical context for decision-making. By using MAPIE, we move beyond simple "0 or 1" predictions to a range of probable outcomes, significantly reducing the risk of false negatives in a medical setting.

## 📥 Installation
To run the notebook locally, clone the repository and install the dependencies:

```bash
pip install dice-ml catboost mapie shap scikit-learn pandas matplotlib seaborn missingno
