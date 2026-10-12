import streamlit as st

st.set_page_config(page_title="Credit Risk & Fraud Detection",
                   page_icon="💳", layout="wide")

st.title("💳 Credit Risk and Fraud Detection")
st.write("A machine-learning system that (1) predicts whether a loan applicant "
         "is likely to default and (2) flags fraudulent card transactions.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Credit Risk")
    st.write("Tuned **Random Forest** on the UCI German Credit dataset "
             "(1,000 applicants), with a cost-based decision threshold.")
    a, b = st.columns(2)
    a.metric("ROC-AUC (test)", "0.793")
    b.metric("Bad borrowers caught", "85%")
    st.page_link("pages/1_credit_risk.py", label="Open Credit Risk Predictor", icon="💳")

with c2:
    st.subheader("Fraud Detection")
    st.write("**XGBoost** on 284,807 card transactions, of which only 0.17% are fraud.")
    a, b = st.columns(2)
    a.metric("PR-AUC", "0.878")
    b.metric("Frauds caught", "84%")
    st.page_link("pages/2_fraud_checker.py", label="Open Fraud Checker", icon="🚨")

st.divider()
l1, l2 = st.columns(2)
l1.page_link("pages/3_data_explorer.py", label="Data Explorer", icon="📊")
l2.page_link("pages/4_model_comparison.py", label="Model Comparison", icon="🏆")

with st.expander("⚠️ Limitations and fairness"):
    st.markdown("""
- **Small, old dataset:** German Credit has only 1,000 records from the 1990s, so results are for learning, not for real lending.
- **Anonymized fraud features:** V1 to V28 are hidden for privacy, so fraud alerts can't be explained in plain business terms.
- **Fairness:** the credit data includes age, sex / marital status and foreign-worker status. Using such attributes in real lending can be discriminatory or illegal. A production system should remove them and test for bias.
- **Correlation, not causation:** the model learns patterns in past data. Some, like "no checking account means lower risk", are dataset quirks.
- **Decision support only:** a human should review every high-risk or fraud flag.
""")