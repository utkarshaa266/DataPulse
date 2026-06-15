import pandas as pd
import os
from datetime import datetime

HISTORY_FILE = "analysis_history.csv"


def save_analysis(
    dataset_name,
    quality_score,
    anomaly_count,
    model_used
):

    new_record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Dataset": dataset_name,
        "Quality Score": quality_score,
        "Anomalies": anomaly_count,
        "Model": model_used
    }])

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        history = pd.concat(
            [history, new_record],
            ignore_index=True
        )

    else:

        history = new_record

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()