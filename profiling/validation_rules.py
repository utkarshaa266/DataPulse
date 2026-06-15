import pandas as pd


def run_validation(df):

    results = []

    # Missing values
    missing = df.isnull().sum().sum()

    if missing > 0:
        results.append(
            f"❌ Found {missing} missing values"
        )
    else:
        results.append(
            "✅ No missing values"
        )

    # Duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        results.append(
            f"❌ Found {duplicates} duplicate rows"
        )
    else:
        results.append(
            "✅ No duplicate rows"
        )

    return results