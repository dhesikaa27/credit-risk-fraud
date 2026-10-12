import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("data/german_credit.csv")
X = df.drop("risk", axis=1)
y = df["risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("models/random_forest_tuned.joblib")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold probabilities on TRAIN data only
proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

# Cost: missing a bad borrower = 5, rejecting a good one = 1
best_t, best_cost = 0.5, float("inf")
for t in np.arange(0.1, 0.9, 0.02):
    tn, fp, fn, tp = confusion_matrix(y_train, proba >= t).ravel()
    cost = 5 * fn + 1 * fp
    if cost < best_cost:
        best_t, best_cost = t, cost
print("Best threshold:", round(best_t, 2))

# Check on the untouched TEST set
test_proba = model.predict_proba(X_test)[:, 1]
pred = (test_proba >= best_t).astype(int)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump({"model": model, "threshold": best_t}, "models/final_model.joblib")