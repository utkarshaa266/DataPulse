import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def detect_anomalies_svm(df):

    result_df = df.copy()

    # Select numeric columns
    numeric_df = df.select_dtypes(include=np.number).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns found. "
            "One-Class SVM requires numeric features."
        )

    # Replace infinity
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

    # Fill missing values using median
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

    # Scale features
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    # One-Class SVM
    model = OneClassSVM(
        kernel="rbf",
        nu=0.05,
        gamma="scale"
    )

    # Predict
    labels = model.fit_predict(
        scaled_data
    )

    # Add results
    result_df["anomaly"] = labels

    return result_df