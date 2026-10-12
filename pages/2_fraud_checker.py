import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Fraud Checker", page_icon="🚨", layout="wide")
st.title("🚨 Fraud Detection")
st.caption("XGBoost model trained on 284,807 card transactions (only 0.17% are fraud).")

THRESHOLD = 0.5
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_model():
    return joblib.load("models/fraud_xgboost.joblib")


@st.cache_data
def load_samples():
    df = pd.read_csv("data/fraud_samples.csv")
    return df.drop("fraud", axis=1), df["fraud"]


model = load_model()
X_s, y_s = load_samples()

tab1, tab2 = st.tabs(["Check a sample transaction", "Upload a CSV"])

with tab1:
    kind = st.radio("Pick a transaction type", ["Normal", "Fraud"], horizontal=True)
    if st.button("Pick a random transaction", type="primary"):
        target = 1 if kind == "Fraud" else 0
        idx = y_s[y_s == target].sample(1).index[0]
        st.session_state["row"] = X_s.loc[[idx], FEATURES]
        st.session_state["actual"] = int(y_s.loc[idx])

    if "row" in st.session_state:
        row = st.session_state["row"]
        p = model.predict_proba(row)[0, 1]
        st.write(f"Transaction amount: **{row['Amount'].iloc[0]:.2f}**")
        st.metric("Fraud probability", f"{p:.1%}")
        st.progress(float(min(p, 1.0)))
        if p >= THRESHOLD:
            st.error("🚨 FRAUD ALERT: block and review this transaction")
        else:
            st.success("✅ Looks like a normal transaction")
        actual = "Fraud" if st.session_state["actual"] == 1 else "Normal"
        st.caption(f"Actual label in the dataset (for demo): {actual}")

with tab2:
    st.write("Upload a CSV with columns V1 to V28 and Amount.")
    frauds = X_s[y_s == 1].sample(5, random_state=1)
    normal = X_s[y_s == 0].sample(95, random_state=1)
    sample = pd.concat([frauds, normal])[FEATURES]
    st.download_button("Download a sample CSV to try", sample.to_csv(index=False),
                       "sample_transactions.csv", "text/csv")

    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["fraud_probability"] = model.predict_proba(data[FEATURES])[:, 1]
            data["flagged"] = data["fraud_probability"] >= THRESHOLD
            st.metric("Flagged as fraud", f"{int(data['flagged'].sum())} of {len(data)}")
            flagged = data[data["flagged"]].sort_values("fraud_probability", ascending=False)
            st.dataframe(flagged[["Amount", "fraud_probability"]])