from ucimlrepo import fetch_ucirepo
import pandas as pd

# Download the German Credit dataset
credit_data = fetch_ucirepo(id=144)

# Get the input features and target labels
X = credit_data.data.features
y = credit_data.data.targets

# Combine them into one table
df = pd.concat([X, y], axis=1)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())