from sklearn.datasets import fetch_openml
import os

print("Downloading... this can take a few minutes, please wait.")
data = fetch_openml(data_id=1597, as_frame=True, parser="auto")
df = data.frame
df = df.rename(columns={"Class": "fraud"})
df["fraud"] = df["fraud"].astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv("data/creditcard.csv", index=False)

print("Shape:", df.shape)
print("\nFraud distribution:")
print(df["fraud"].value_counts())
print("\nFraud percentage:", round(df["fraud"].mean() * 100, 3), "%")
print("\nColumns:", df.columns.tolist())