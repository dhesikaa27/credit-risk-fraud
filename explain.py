import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/german_credit.csv")
X = df.drop("risk", axis=1)
y = df["risk"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

saved = joblib.load("models/final_model.joblib")
pipe = saved["model"]
pre = pipe.named_steps["pre"]
rf = pipe.named_steps["model"]

Xt = pre.transform(X_test)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
Xt = pd.DataFrame(Xt, columns=names)

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Xt)
if isinstance(sv, list):
    sv = sv[1]
elif sv.ndim == 3:
    sv = sv[:, :, 1]

plt.figure()
shap.summary_plot(sv, Xt, show=False, max_display=15)
plt.tight_layout()
plt.savefig("data/shap_summary.png", bbox_inches="tight")
plt.show()

plt.figure()
shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig("data/shap_bar.png", bbox_inches="tight")
plt.show()