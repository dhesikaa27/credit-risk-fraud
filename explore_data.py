from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import os

credit_data = fetch_ucirepo(id=144)
df = pd.concat([credit_data.data.features, credit_data.data.targets], axis=1)

# Give columns meaningful names
df.columns = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "risk"
]

# 1 = good, 2 = bad  ->  0 = good, 1 = bad (default)
df["risk"] = df["risk"].map({1: 0, 2: 1})

# Save a local copy so we don't need internet again
os.makedirs("data", exist_ok=True)
df.to_csv("data/german_credit.csv", index=False)

print("Shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nRisk distribution:")
print(df["risk"].value_counts())
print("\nNumeric summary:")
print(df.describe())

# Chart: good vs bad
df["risk"].value_counts().plot(kind="bar", color=["green", "red"])
plt.title("Good (0) vs Bad (1) Credit")
plt.savefig("data/risk_distribution.png")
plt.show()