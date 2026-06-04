from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def detect_anomalies_svm(df):

    numeric_df = df.select_dtypes(
        include=['float64', 'int64']
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        numeric_df
    )

    model = OneClassSVM(
        kernel='rbf',
        nu=0.05
    )

    labels = model.fit_predict(
        scaled_data
    )

    result_df = df.copy()

    result_df['anomaly'] = labels

    return result_df