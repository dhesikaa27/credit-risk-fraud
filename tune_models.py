import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv("data/german_credit.csv")
X = df.drop("risk", axis=1)
y = df["risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42),
}

print("5-FOLD CROSS-VALIDATION (ROC-AUC)")
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Tune Random Forest
print("\nTuning Random Forest...")
rf_pipe = Pipeline([("pre", pre), ("model", models["Random Forest"])])
grid = GridSearchCV(
    rf_pipe,
    {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3, 5],
    },
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 3))

best = grid.best_estimator_
proba = best.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print(classification_report(y_test, best.predict(X_test)))

joblib.dump(best, "models/random_forest_tuned.joblib")