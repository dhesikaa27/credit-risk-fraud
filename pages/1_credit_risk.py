import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Predictor")
st.caption("Fill in the applicant's details or load an example. The result updates instantly.")

saved = joblib.load("models/final_model.joblib")
model, threshold = saved["model"], saved["threshold"]

LABELS = {
    "checking_account": {"A11": "Below 0 DM", "A12": "0 to 200 DM",
        "A13": "200 DM or more", "A14": "No checking account"},
    "credit_history": {"A30": "No credits taken / all paid back",
        "A31": "All credits at this bank paid back",
        "A32": "Existing credits paid back on time",
        "A33": "Delay in paying in the past",
        "A34": "Critical account / credits elsewhere"},
    "purpose": {"A40": "New car", "A41": "Used car", "A42": "Furniture / equipment",
        "A43": "Radio / TV", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other"},
    "savings_account": {"A61": "Below 100 DM", "A62": "100 to 500 DM",
        "A63": "500 to 1000 DM", "A64": "1000 DM or more", "A65": "Unknown / none"},
    "employment_since": {"A71": "Unemployed", "A72": "Less than 1 year",
        "A73": "1 to 4 years", "A74": "4 to 7 years", "A75": "7 years or more"},
    "personal_status_sex": {"A91": "Male: divorced / separated",
        "A92": "Female: divorced / separated / married",
        "A93": "Male: single", "A94": "Male: married / widowed", "A95": "Female: single"},
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {"A121": "Real estate", "A122": "Savings agreement / life insurance",
        "A123": "Car or other", "A124": "Unknown / no property"},
    "other_installments": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "Free"},
    "job": {"A171": "Unemployed / unskilled non-resident", "A172": "Unskilled resident",
        "A173": "Skilled employee", "A174": "Management / highly qualified"},
    "telephone": {"A191": "None", "A192": "Yes"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}

NUMERIC = {  # name: (min, max, default, step)
    "duration_months": (4, 72, 18, 1),
    "credit_amount": (250, 20000, 2300, 50),
    "installment_rate": (1, 4, 3, 1),
    "residence_since": (1, 4, 2, 1),
    "age": (18, 75, 35, 1),
    "existing_credits": (1, 4, 1, 1),
    "dependents": (1, 2, 1, 1),
}

HELP = {
    "duration_months": "Length of the loan in months. Longer loans are usually riskier.",
    "credit_amount": "Loan amount requested (in DM, the currency of the dataset).",
    "installment_rate": "Instalment as a share of income (1 = low, 4 = high).",
    "checking_account": "Balance in the applicant's checking account.",
    "savings_account": "Savings balance. Low savings means higher risk.",
    "credit_history": "How the applicant repaid past credits.",
    "employment_since": "How long the applicant has been in the current job.",
    "existing_credits": "Number of existing credits at this bank.",
    "other_installments": "Other instalment plans the applicant already has.",
    "property": "Most valuable property the applicant owns.",
}

GROUPS = {
    "💰 Loan details": ["duration_months", "credit_amount", "purpose", "installment_rate"],
    "🏦 Financial profile": ["checking_account", "savings_account", "credit_history",
                            "existing_credits", "other_installments", "property"],
    "👤 Personal details": ["age", "employment_since", "job", "housing", "residence_since",
                           "dependents", "personal_status_sex", "other_debtors",
                           "telephone", "foreign_worker"],
}
ORDER = [c for g in GROUPS.values() for c in g]

DEFAULTS = {c: (NUMERIC[c][2] if c in NUMERIC else list(LABELS[c])[0]) for c in ORDER}

RISKY = {"checking_account": "A11", "duration_months": 48, "credit_history": "A33",
    "purpose": "A43", "credit_amount": 9000, "savings_account": "A61",
    "employment_since": "A72", "installment_rate": 4, "personal_status_sex": "A93",
    "other_debtors": "A101", "residence_since": 2, "property": "A124", "age": 24,
    "other_installments": "A141", "housing": "A151", "existing_credits": 2,
    "job": "A173", "dependents": 1, "telephone": "A191", "foreign_worker": "A201"}

SAFE = {"checking_account": "A14", "duration_months": 12, "credit_history": "A32",
    "purpose": "A43", "credit_amount": 1500, "savings_account": "A64",
    "employment_since": "A75", "installment_rate": 2, "personal_status_sex": "A94",
    "other_debtors": "A101", "residence_since": 4, "property": "A121", "age": 45,
    "other_installments": "A143", "housing": "A152", "existing_credits": 1,
    "job": "A173", "dependents": 1, "telephone": "A192", "foreign_worker": "A201"}


def load_preset(values):
    for k, v in values.items():
        st.session_state[k] = v


def pretty(name):
    for col, mapping in LABELS.items():
        if name.startswith(col + "_"):
            code = name[len(col) + 1:]
            return f"{col.replace('_', ' ').title()}: {mapping.get(code, code)}"
    return name.replace("_", " ").title()


@st.cache_resource
def get_explainer():
    pre = model.named_steps["pre"]
    rf = model.named_steps["model"]
    return pre, shap.TreeExplainer(rf)


# ---------- Example buttons ----------
b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 3])
b1.button("🟢 Safe applicant example", on_click=load_preset, args=(SAFE,))
b2.button("🔴 Risky applicant example", on_click=load_preset, args=(RISKY,))
b3.button("↺ Reset", on_click=load_preset, args=(DEFAULTS,))

# ---------- Inputs ----------
inputs = {}
tabs = st.tabs(list(GROUPS.keys()))
for tab, cols_list in zip(tabs, GROUPS.values()):
    with tab:
        cols = st.columns(2)
        for i, c in enumerate(cols_list):
            title = c.replace("_", " ").title()
            with cols[i % 2]:
                if c in LABELS:
                    inputs[c] = st.selectbox(title, list(LABELS[c].keys()),
                        format_func=lambda k, c=c: LABELS[c][k],
                        key=c, help=HELP.get(c))
                else:
                    lo, hi, default, step = NUMERIC[c]
                    inputs[c] = st.slider(title, lo, hi, default, step,
                                          key=c, help=HELP.get(c))

# ---------- Prediction ----------
row = pd.DataFrame([inputs])
p = float(model.predict_proba(row)[0, 1])

if p >= threshold:
    level, color, advice = "HIGH RISK", "#d64545", "Recommend rejection or manual review"
elif p >= threshold * 0.6:
    level, color, advice = "MEDIUM RISK", "#e0a800", "Borderline case: review manually"
else:
    level, color, advice = "LOW RISK", "#2e9e5b", "Recommend approval"

st.divider()
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Result")
    st.markdown(
        f"<div style='background:{color};color:white;padding:14px 18px;border-radius:12px;"
        f"font-size:22px;font-weight:700;'>{level}</div>", unsafe_allow_html=True)
    st.metric("Default probability", f"{p:.1%}")
    st.write(advice)

    pos = min(max(p * 100, 0.5), 99.5)
    st.markdown(f"""
<div style="position:relative;height:26px;border-radius:13px;
background:linear-gradient(90deg,#2e9e5b 0%,#f2c744 40%,#d64545 100%);margin:12px 0 6px 0;">
<div style="position:absolute;left:{pos:.1f}%;top:-6px;width:4px;height:38px;
background:#1B2A41;border-radius:2px;"></div>
<div style="position:absolute;left:{threshold*100:.0f}%;top:0;width:2px;height:26px;
background:white;opacity:.9;"></div></div>
<div style="display:flex;justify-content:space-between;font-size:12px;color:#667;">
<span>0% low</span><span>white line = decision threshold ({threshold:.0%})</span>
<span>100% high</span></div>
""", unsafe_allow_html=True)

    report = {c.replace("_", " ").title(): (LABELS[c][inputs[c]] if c in LABELS else inputs[c])
              for c in ORDER}
    report["Default probability"] = f"{p:.1%}"
    report["Decision"] = level
    st.download_button("⬇️ Download result (CSV)",
                       pd.DataFrame([report]).T.reset_index().to_csv(index=False, header=["Item", "Value"]),
                       "credit_risk_result.csv", "text/csv")

with right:
    st.subheader("Why this result?")
    try:
        pre, explainer = get_explainer()
        x = pre.transform(row)
        if hasattr(x, "toarray"):
            x = x.toarray()
        sv = explainer.shap_values(x)
        if isinstance(sv, list):
            sv = np.array(sv[1])[0]
        else:
            sv = np.array(sv)
            sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        names = [pretty(n.split("__", 1)[1]) for n in pre.get_feature_names_out()]
        exp = pd.DataFrame({"Factor": names, "Impact": sv})
        exp = exp[x[0] != 0]
        exp = exp.reindex(exp["Impact"].abs().sort_values(ascending=False).index).head(8)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        cols_bar = ["#d64545" if v > 0 else "#2e9e5b" for v in exp["Impact"]]
        ax.barh(exp["Factor"].tolist()[::-1], exp["Impact"].tolist()[::-1],
                color=cols_bar[::-1])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("← lowers risk        raises risk →")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Red bars push toward HIGH risk. Green bars push toward LOW risk.")
    except Exception as e:
        st.info(f"Explanation not available: {e}")