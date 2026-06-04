from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def detect_anomalies_dbscan(df):

    numeric_df = df.select_dtypes(
        include=['float64', 'int64']
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    model = DBSCAN(
        eps=0.5,
        min_samples=5
    )

    labels = model.fit_predict(
        scaled_data
    )

    result_df = df.copy()

    result_df['anomaly'] = labels

    return result_df