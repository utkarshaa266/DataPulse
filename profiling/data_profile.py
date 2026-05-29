import pandas as pd

print("Loading dataset...")

# Load dataset
df = pd.read_csv(
    "data/raw/creditcard.csv",
    nrows=10000
)

print("Dataset Loaded Successfully!")

# -----------------------------
# BASIC INFORMATION
# -----------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# MISSING VALUES
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Null percentage
null_percentage = (
    df.isnull().sum() / len(df)
) * 100

print("\nNull Percentage:")
print(null_percentage)

# -----------------------------
# DUPLICATES
# -----------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicate_count)

# -----------------------------
# STATISTICAL SUMMARY
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())