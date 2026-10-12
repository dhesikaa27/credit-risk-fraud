import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")
X = df.drop("fraud", axis=1)
y = df["fraud"]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

test = X_test.copy()
test["fraud"] = y_test
sample = pd.concat([
    test[test["fraud"] == 1],
    test[test["fraud"] == 0].sample(2000, random_state=1),
])
sample.to_csv("data/fraud_samples.csv", index=False)
print("Saved", sample.shape, "rows to data/fraud_samples.csv")