def generate_insights(df, anomaly_count):

    insights = []

    # Missing values insight
    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        insights.append(
            f"Dataset contains {missing_values} missing values."
        )
    else:
        insights.append(
            "No missing values detected."
        )

    # Duplicate rows insight
    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        insights.append(
            f"Dataset contains {duplicate_rows} duplicate rows."
        )
    else:
        insights.append(
            "No duplicate rows detected."
        )

    # Anomaly insight
    if anomaly_count > 0:
        insights.append(
            f"{anomaly_count} anomalies were detected in the dataset."
        )
    else:
        insights.append(
            "No anomalies detected."
        )

    return insights