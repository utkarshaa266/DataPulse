import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):

    result_df = df.copy()

    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns found. "
            "Isolation Forest requires at least one numeric column."
        )

    # Replace infinite values
    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns containing only NaN
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.empty:
        raise ValueError(
            "No usable numeric columns found after cleaning."
        )

    # Fill missing values with median
    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    # Remove constant columns
    numeric_df = numeric_df.loc[
        :, numeric_df.nunique() > 1
    ]

    if numeric_df.empty:
        raise ValueError(
            "Dataset contains no useful numeric variation."
        )

    # Create model
    model = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100
    )

    # Train model
    model.fit(numeric_df)

    # Predict anomalies
    predictions = model.predict(numeric_df)

    # Add predictions
    result_df["anomaly"] = predictions

    return result_df