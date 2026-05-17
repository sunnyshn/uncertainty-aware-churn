# Uncertainty-Aware Churn Prediction
***

Bayesian neural network for customer churn prediction that outputs **calibrated probabilities with uncertainty**, enabling cost-sensitive routing of retention spend by confidence tier.

## Why?:

Standard churn models output a single probability per customer. Retention teams need more: *how confident is the model?* High-probability + high-confidence customers warrant personalized outreach (expensive); high-probability + low-confidence customers are better routed to automated outreach (cheap). This project quantifies that confidence and demonstrates the cost savings empirically.

## Approach

| Stage | Method |
|---|---|
| Baselines | Logistic Regression, XGBoost |
| Main model | Bayesian Neural Network via Monte Carlo Dropout (PyTorch) |
| Calibration | Reliability diagrams, Expected Calibration Error |
| Evaluation | Cost-sensitive policy simulation (uncertainty-routed vs. naive) |

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7K customers, 20 features)

## Status

In progress. 

## Build phases

1. Setup & EDA
2. Baselines (Logistic Regression + XGBoost)
3. Bayesian NN (MC Dropout in PyTorch) - *current*
4. Calibration + cost-sensitive evaluation
5. Packaging (Docker, README polish)
6. Exec brief (1-page PDF)

***
### Early EDA findings:
- Churn distribution is moderately imbalanced, with a churn percentage rate of 26.5% amongst customers. Will need to rely on AUC/precision/recall for primary metrics. 
- Churners have substantially shorter tenure, with a median difference of ~20 months, and a slightly higher monthly cost, and lower total cost, exhibited in early-life churn. 

## Baseline results
Train/test split: 80/20, stratified on `Churn`. Threshold = 0.5 for class predictions.

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.858 | 0.811 | 0.664 | 0.586 | 0.622 |
| XGBoost (untuned) | 0.832 | 0.784 | 0.605 | 0.532 | 0.566 |

Both models trained on the same one-hot-encoded + standard-scaled feature matrix. Logistic Regression outperforms an untuned XGBoost on this dataset, consistent with Telco's largely linear churn signal at this sample size. These serve as references for the Bayesian neural network in Phase 3.
