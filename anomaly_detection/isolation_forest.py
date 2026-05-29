import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):

    # Select only numeric columns
    numeric_df = df.select_dtypes(
        include=['float64', 'int64']
    )

    # Create model
    model = IsolationForest(
        contamination=0.01,
        random_state=42
    )

    # Train model
    model.fit(numeric_df)

    # Predict anomalies
    predictions = model.predict(numeric_df)

    # Add predictions to dataframe
    df['anomaly'] = predictions

    return df