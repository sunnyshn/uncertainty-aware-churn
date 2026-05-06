# Uncertainty-Aware Churn Prediction

Bayesian neural network for customer churn prediction that outputs **calibrated probabilities with uncertainty**, enabling cost-sensitive routing of retention spend by confidence tier.

## Why

Standard churn models output a single probability per customer. Retention teams need more: *how confident is the model?* High-probability + high-confidence customers warrant personalized outreach (expensive); high-probability + low-confidence customers are better routed to automated outreach (cheap). This project quantifies that confidence and demonstrates the cost savings empirically.

## Approach

| Stage | Method |
|---|---|
| Baselines | Logistic Regression, XGBoost |
| Main model | Bayesian Neural Network via Monte Carlo Dropout (PyTorch) |
| Calibration | Reliability diagrams, Expected Calibration Error |
| Evaluation | Cost-sensitive policy simulation (uncertainty-routed vs. naive) |

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7K customers, 20 features). Place the CSV at `data/telco_churn.csv`.

## Status

In progress. Phase 1 (setup + EDA) underway.

## Build phases

1. Setup & EDA — *current*
2. Baselines (Logistic Regression + XGBoost)
3. Bayesian NN (MC Dropout in PyTorch)
4. Calibration + cost-sensitive evaluation
5. Packaging (Docker, README polish)
6. Exec brief (1-page PDF)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```
