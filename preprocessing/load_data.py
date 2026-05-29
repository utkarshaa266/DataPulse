import pandas as pd

print("Loading dataset...")

# Load only first 10,000 rows initially
df = pd.read_csv(
    "data/raw/creditcard.csv",
    nrows=10000
)

print("Dataset loaded successfully!")

# Show first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())