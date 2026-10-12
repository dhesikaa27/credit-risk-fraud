# Credit Risk and Fraud Detection

A machine-learning project with two modules and a Streamlit dashboard:
1. **Credit risk**: predicts whether a loan applicant will default.
2. **Fraud detection**: flags fraudulent credit card transactions.

## Datasets
- UCI German Credit (1,000 applicants, 20 features)
- Credit card fraud (284,807 transactions, 0.17% fraud)

## Models and results
| Module | Final model | Key result |
|---|---|---|
| Credit risk | Tuned Random Forest, cost-based threshold 0.4 | ROC-AUC 0.793, catches 85% of bad borrowers |
| Fraud | XGBoost | PR-AUC 0.878, 82 of 98 frauds caught, 11 false alarms |

Other methods: Logistic Regression baselines, 5-fold cross-validation,
GridSearchCV, class weighting, SHAP explainability.

## How to run
1. Create a virtual environment and activate it
2. `pip install -r requirements.txt`
3. `python explore_data.py`, then `python train_model.py`, `python tune_models.py`, `python threshold.py`
4. `python load_fraud.py`, `python train_fraud.py`, `python make_fraud_samples.py`
5. `python explain.py`
6. `streamlit run Home.py`

## Project structure
- `Home.py`: dashboard home page
- `pages/`: Credit Risk, Fraud Checker, Data Explorer, Model Comparison
- `data/`: datasets and charts
- `models/`: saved models

## Limitations
Small, old credit dataset. Anonymized fraud features. The credit data contains
age, sex and foreign-worker fields, which need bias checks before any real use.
Decision support only.