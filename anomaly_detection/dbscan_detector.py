import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def detect_anomalies_dbscan(df):

    result_df = df.copy()

    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns found. "
            "DBSCAN requires numeric features."
        )

    # Replace infinite values
    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove completely empty columns
    numeric_df = numeric_df.dropna(
        axis=1,
        how="all"
    )

    if numeric_df.empty:
        raise ValueError(
            "No usable numeric columns found after cleaning."
        )

    # Fill missing values
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

    # Standardize features
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    # DBSCAN model
    model = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    labels = model.fit_predict(
        scaled_data
    )

    # DBSCAN uses -1 for anomalies/noise
    result_df["anomaly"] = labels

    return result_df