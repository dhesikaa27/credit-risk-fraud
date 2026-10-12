import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")

st.header("Credit risk models")
credit = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost", "Random Forest", "Random Forest (tuned)"],
    "CV ROC-AUC": [0.769, 0.782, 0.788, 0.798],
})
st.dataframe(credit, hide_index=True)
st.success("Chosen model: tuned Random Forest (best CV ROC-AUC, test ROC-AUC 0.793).")

st.subheader("Decision threshold tuning")
thr = pd.DataFrame({
    "Threshold": ["0.5 (default)", "0.4 (cost-tuned)"],
    "Bad borrowers caught (of 60)": [39, 51],
    "Bad borrowers missed": [21, 9],
})
st.dataframe(thr, hide_index=True)
st.caption("Missing a bad borrower costs about 5 times more than rejecting a good one, "
           "so we lowered the threshold.")

st.header("Fraud models")
fraud = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost"],
    "ROC-AUC": [0.971, 0.969],
    "PR-AUC": [0.720, 0.878],
    "Frauds caught (of 98)": [90, 82],
    "False alarms": [1439, 11],
    "Precision": ["5.9%", "88.2%"],
})
st.dataframe(fraud, hide_index=True)
st.success("Chosen model: XGBoost (far fewer false alarms and the best PR-AUC).")

st.header("Explainability (SHAP)")
if os.path.exists("data/shap_summary.png"):
    st.image("data/shap_summary.png", caption="How each factor pushes risk up or down")
if os.path.exists("data/shap_bar.png"):
    st.image("data/shap_bar.png", caption="Average importance of each factor")