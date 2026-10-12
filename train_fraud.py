import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")
X = df.drop("fraud", axis=1)
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pre = ColumnTransformer([("amt", StandardScaler(), ["Amount"])],
                        remainder="passthrough")

weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=weight, tree_method="hist",
        eval_metric="aucpr", n_jobs=-1, random_state=42,
    ),
}

os.makedirs("models", exist_ok=True)

for name, clf in models.items():
    print("=" * 50)
    print("Training", name, "...")
    pipe = Pipeline([("pre", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=3))

    joblib.dump(pipe, f"models/fraud_{name.replace(' ', '_').lower()}.joblib")