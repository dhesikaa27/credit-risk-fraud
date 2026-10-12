import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")

df = pd.read_csv("data/german_credit.csv")

st.header("Credit risk data (German Credit)")
c1, c2, c3 = st.columns(3)
c1.metric("Applicants", f"{len(df):,}")
c2.metric("Bad credit", f"{df['risk'].mean():.0%}")
c3.metric("Average loan amount", f"{df['credit_amount'].mean():,.0f}")

st.subheader("Good vs bad credit")
st.bar_chart(df["risk"].map({0: "Good", 1: "Bad"}).value_counts())

st.subheader("Bad-credit rate by feature")
st.caption("Pick a feature to see which groups default more often.")
numeric = ["duration_months", "credit_amount", "age"]
categorical = [c for c in df.columns if c not in numeric + ["risk"]
               and df[c].dtype == "object"]
col = st.selectbox("Feature", numeric + categorical)

if col in numeric:
    groups = pd.qcut(df[col], 6, duplicates="drop").astype(str)
    rate = df.groupby(groups)["risk"].mean()
else:
    rate = df.groupby(col)["risk"].mean()
st.bar_chart(rate)

with st.expander("View raw data"):
    st.dataframe(df.head(100))

st.header("Fraud data (credit card transactions)")
f1, f2, f3 = st.columns(3)
f1.metric("Transactions", "284,807")
f2.metric("Frauds", "492")
f3.metric("Fraud rate", "0.173%")
st.info("Fraud is extremely rare, about 1 in 600. That is why we judge the fraud "
        "model by recall, precision and PR-AUC instead of accuracy.")