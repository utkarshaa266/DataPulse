
# ============================================================
# DATAPULSE
# AI-POWERED DATA QUALITY & ANOMALY DETECTION PLATFORM
# ============================================================
from sklearn.datasets import make_classification, make_regression
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from streamlit_autorefresh import st_autorefresh

# ============================================================
# EXISTING MODULES
# ============================================================

from anomaly_detection.dbscan_detector import (
    detect_anomalies_dbscan
)

from anomaly_detection.one_class_svm import (
    detect_anomalies_svm
)

from anomaly_detection.isolation_forest import (
    detect_anomalies
)

from visualizations.charts import (
    create_histogram,
    create_boxplot,
    create_scatter,
    create_heatmap
)

from profiling.validation_rules import (
    run_validation
)

from profiling.quality_score import (
    calculate_quality_score
)

from storage.history_manager import (
    save_analysis,
    load_history
)

from insights.ai_insights import (
    generate_insights
)


# ============================================================
# MACHINE LEARNING IMPORTS
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DataPulse",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 DataPulse")

page = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Dataset Workflow",
        "Dashboard",
        "Anomaly Detection",
        "Data Cleaning",
        "Visualizations",
        "Real-Time Monitoring",
        "Analysis History"
    ]
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 DataPulse")

st.subheader(
    "AI-Powered Data Quality & Anomaly Detection Platform"
)

# ============================================================
# SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_dataset(
    dataset_type,
    rows,
    features,
    anomaly_percentage=5,
    missing_percentage=0,
    duplicate_percentage=0,
    outlier_percentage=0,
    noise_level=0.1
):

    try:

        # ----------------------------------------------------
        # CLASSIFICATION DATASET
        # ----------------------------------------------------

        if dataset_type == "Classification":

            X, y = make_classification(
                n_samples=rows,
                n_features=features,
                n_informative=max(2, int(features * 0.6)),
                n_redundant=0,
                n_repeated=0,
                n_classes=2,
                random_state=42
            )

            columns = [
                f"Feature_{i+1}"
                for i in range(features)
            ]

            df = pd.DataFrame(
                X,
                columns=columns
            )

            df["Target"] = y


        # ----------------------------------------------------
        # REGRESSION DATASET
        # ----------------------------------------------------

        elif dataset_type == "Regression":

            X, y = make_regression(
                n_samples=rows,
                n_features=features,
                n_informative=max(
                    2,
                    min(
                        features,
                        int(features * 0.7)
                    )
                ),
                noise=noise_level * 10,
                random_state=42
            )

            columns = [
                f"Feature_{i+1}"
                for i in range(features)
            ]

            df = pd.DataFrame(
                X,
                columns=columns
            )

            df["Target"] = y


        # ----------------------------------------------------
        # ANOMALY DETECTION DATASET
        # ----------------------------------------------------

        elif dataset_type == "Anomaly Detection":

            X = np.random.normal(
                loc=0,
                scale=1,
                size=(rows, features)
            )

            df = pd.DataFrame(
                X,
                columns=[
                    f"Feature_{i+1}"
                    for i in range(features)
                ]
            )

            # Inject anomalies
            anomaly_count = int(
                rows * anomaly_percentage / 100
            )

            if anomaly_count > 0:

                anomaly_indices = np.random.choice(
                    rows,
                    anomaly_count,
                    replace=False
                )

                for index in anomaly_indices:

                    df.iloc[index] = (
                        df.iloc[index] * 6
                    )


        # ----------------------------------------------------
        # MIXED DATASET
        # ----------------------------------------------------

        elif dataset_type == "Mixed Dataset":

            numeric_features = max(
                2,
                features - 2
            )

            X = np.random.normal(
                loc=5000,
                scale=500,
                size=(rows, numeric_features)
            )

            df = pd.DataFrame(
                X,
                columns=[
                    f"Numeric_{i+1}"
                    for i in range(numeric_features)
                ]
            )

            df["Category"] = np.random.choice(
                ["A", "B", "C"],
                rows
            )

            df["Status"] = np.random.choice(
                ["Active", "Inactive"],
                rows
            )


        else:

            raise ValueError(
                "Invalid dataset type selected."
            )


        # ----------------------------------------------------
        # ADD NOISE
        # ----------------------------------------------------

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        target_column = (
            "Target"
            if "Target" in numeric_columns
            else None
        )

        feature_columns = [
            col
            for col in numeric_columns
            if col != target_column
        ]

        if noise_level > 0 and feature_columns:

            noise = np.random.normal(
                0,
                noise_level,
                size=(
                    len(df),
                    len(feature_columns)
                )
            )

            df[feature_columns] = (
                df[feature_columns] + noise
            )


        # ----------------------------------------------------
        # INJECT OUTLIERS
        # ----------------------------------------------------

        if (
            outlier_percentage > 0
            and feature_columns
        ):

            outlier_count = int(
                rows * outlier_percentage / 100
            )

            if outlier_count > 0:

                outlier_indices = np.random.choice(
                    len(df),
                    min(
                        outlier_count,
                        len(df)
                    ),
                    replace=False
                )

                for col in feature_columns:

                    df.loc[
                        outlier_indices,
                        col
                    ] *= 8


        # ----------------------------------------------------
        # INJECT MISSING VALUES
        # ----------------------------------------------------

        if (
            missing_percentage > 0
            and len(df) > 0
        ):

            total_cells = (
                len(df) * len(df.columns)
            )

            missing_count = int(
                total_cells *
                missing_percentage /
                100
            )

            if missing_count > 0:

                for _ in range(missing_count):

                    row_index = np.random.randint(
                        0,
                        len(df)
                    )

                    column_index = np.random.randint(
                        0,
                        len(df.columns)
                    )

                    df.iat[
                        row_index,
                        column_index
                    ] = np.nan


        # ----------------------------------------------------
        # ADD DUPLICATES
        # ----------------------------------------------------

        if (
            duplicate_percentage > 0
            and len(df) > 0
        ):

            duplicate_count = int(
                len(df) *
                duplicate_percentage /
                100
            )

            if duplicate_count > 0:

                duplicate_rows = df.sample(
                    n=min(
                        duplicate_count,
                        len(df)
                    ),
                    random_state=42
                )

                df = pd.concat(
                    [
                        df,
                        duplicate_rows
                    ],
                    ignore_index=True
                )


        # ----------------------------------------------------
        # SHUFFLE DATASET
        # ----------------------------------------------------

        df = df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)


        return df


    except Exception as e:

        raise RuntimeError(
            f"Synthetic dataset generation failed: {e}"
        )
# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

df = None

if uploaded_file is not None:

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file
        )

        # New upload replaces previously generated dataset
        if "generated_dataset" in st.session_state:
            del st.session_state["generated_dataset"]

    except pd.errors.EmptyDataError:

        st.error(
            "❌ The uploaded CSV is empty."
        )
        st.stop()

    except pd.errors.ParserError:

        st.error(
            "❌ The CSV format is invalid or corrupted."
        )
        st.stop()

    except Exception as e:

        st.error(
            f"❌ Unable to read CSV: {e}"
        )
        st.stop()


# ============================================================
# USE GENERATED DATASET
# ============================================================

if (
    df is None
    and page == "🚀 Dataset Workflow"
    and "generated_dataset" in st.session_state
):

    df = st.session_state[
        "generated_dataset"
    ]

# ============================================================
# SYNTHETIC DATASET GENERATOR
# ============================================================

if page == "🚀 Dataset Workflow":

    st.sidebar.divider()

    st.sidebar.subheader(
        "🧬 Dataset Source"
    )

    dataset_source = st.sidebar.radio(
        "Choose dataset source",
        [
            "Upload CSV",
            "Generate Synthetic Dataset"
        ],
        key="dataset_source"
    )

    if dataset_source == "Generate Synthetic Dataset":

        st.header(
            "🧬 Smart Synthetic Dataset Generator"
        )

        st.write(
            "Generate controlled datasets with "
            "realistic data-quality problems for "
            "testing and demonstrating DataPulse."
        )

        col1, col2 = st.columns(2)

        with col1:

            synthetic_type = st.selectbox(
                "Dataset Type",
                [
                    "Classification",
                    "Regression",
                    "Anomaly Detection",
                    "Mixed Dataset"
                ]
            )

            synthetic_rows = st.slider(
                "Number of Rows",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100
            )

            synthetic_features = st.slider(
                "Number of Features",
                min_value=2,
                max_value=15,
                value=8
            )

        with col2:

            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05
            )

            anomaly_percentage = st.slider(
                "Anomaly Percentage",
                min_value=0,
                max_value=20,
                value=5,
                step=1
            )

            missing_percentage = st.slider(
                "Missing Value Percentage",
                min_value=0,
                max_value=20,
                value=5,
                step=1
            )

            duplicate_percentage = st.slider(
                "Duplicate Percentage",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )

            outlier_percentage = st.slider(
                "Outlier Percentage",
                min_value=0,
                max_value=10,
                value=3,
                step=1
            )

        st.info(
            "💡 DataPulse will intentionally introduce "
            "data-quality issues so you can demonstrate "
            "cleaning and anomaly detection."
        )

        if st.button(
            "🧬 Generate Dataset",
            type="primary",
            use_container_width=True
        ):

            try:

                generated_df = (
                    generate_synthetic_dataset(
                        dataset_type=synthetic_type,
                        rows=synthetic_rows,
                        features=synthetic_features,
                        anomaly_percentage=anomaly_percentage,
                        missing_percentage=missing_percentage,
                        duplicate_percentage=duplicate_percentage,
                        outlier_percentage=outlier_percentage,
                        noise_level=noise_level
                    )
                )

                st.session_state[
                    "generated_dataset"
                ] = generated_df

                st.success(
                    "✅ Synthetic dataset generated successfully!"
                )

            except Exception as e:

                st.error(
                    f"❌ Dataset generation failed: {e}"
                )

    # Use generated dataset
    if "generated_dataset" in st.session_state:

        df = st.session_state[
            "generated_dataset"
        ]

        st.divider()

        st.subheader(
            "📊 Generated Dataset"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Missing Values",
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        generated_csv = (
            df.to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Generated Dataset",
            generated_csv,
            "DataPulse_synthetic_dataset.csv",
            "text/csv",
            key="synthetic_download"
        )
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_numeric_columns(data):

    return data.select_dtypes(
        include=np.number
    ).columns.tolist()


def get_categorical_columns(data):

    return data.select_dtypes(
        exclude=np.number
    ).columns.tolist()


def get_anomaly_column(data):

    if "Anomaly" in data.columns:

        return "Anomaly"

    if "anomaly" in data.columns:

        return "anomaly"

    return None


def dataset_scan(data):

    numeric_columns = get_numeric_columns(
        data
    )

    categorical_columns = get_categorical_columns(
        data
    )

    missing_values = int(
        data.isnull().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    infinite_values = int(
        np.isinf(
            data.select_dtypes(
                include=np.number
            )
        ).sum().sum()
    )

    constant_columns = [
        column
        for column in data.columns
        if data[column].nunique(
            dropna=False
        ) <= 1
    ]

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "numeric": len(numeric_columns),
        "categorical": len(categorical_columns),
        "missing": missing_values,
        "duplicates": duplicate_rows,
        "infinite": infinite_values,
        "constant": len(constant_columns),
        "constant_columns": constant_columns
    }


def clean_dataset(data):

    cleaned = data.copy()

    # Replace infinity
    cleaned = cleaned.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_columns = get_numeric_columns(
        cleaned
    )

    categorical_columns = get_categorical_columns(
        cleaned
    )

    # Numeric → median
    for column in numeric_columns:

        if cleaned[column].isnull().any():

            median_value = cleaned[
                column
            ].median()

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

    # Categorical → mode
    for column in categorical_columns:

        if cleaned[column].isnull().any():

            mode_values = cleaned[
                column
            ].mode()

            if not mode_values.empty:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode_values.iloc[0])
                )

    # Remove duplicates
    cleaned = (
        cleaned
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return cleaned


def find_classification_target(data):

    categorical_columns = get_categorical_columns(
        data
    )

    # Prefer categorical columns with reasonable
    # number of classes
    for column in categorical_columns:

        unique_count = data[
            column
        ].nunique()

        if 2 <= unique_count <= 20:

            return column

    # Numeric binary target
    for column in get_numeric_columns(data):

        unique_count = data[
            column
        ].nunique()

        if unique_count == 2:

            return column

    return None


def find_regression_target(data):

    numeric_columns = get_numeric_columns(
        data
    )

    # Prefer numeric columns with multiple
    # unique values
    for column in reversed(
        numeric_columns
    ):

        if data[column].nunique() > 10:

            return column

    return None

# ============================================================
# MODEL COMPARISON CHARTS
# ============================================================

def show_model_comparison_charts(
    results_df,
    problem_type
):

    if results_df is None or results_df.empty:
        return

    st.subheader("📊 Model Performance Comparison")

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if problem_type == "Classification":

        # -----------------------------------------------
        # PERFORMANCE METRICS
        # -----------------------------------------------

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        available_metrics = [
            metric
            for metric in metric_columns
            if metric in results_df.columns
        ]

        if available_metrics:

            chart_data = results_df[
                ["Model"] + available_metrics
            ].copy()

            chart_data = chart_data.set_index(
                "Model"
            )

            st.bar_chart(
                chart_data,
                use_container_width=True
            )

        # -----------------------------------------------
        # BEST MODEL
        # -----------------------------------------------

        if "F1 Score" in results_df.columns:

            best_row = results_df.iloc[0]

            st.success(
                f"🏆 Best Classification Model: "
                f"**{best_row['Model']}** "
                f"(F1 Score: {best_row['F1 Score']:.4f})"
            )
            


    # ========================================================
    # REGRESSION
    # ========================================================

    elif problem_type == "Regression":

        # -----------------------------------------------
        # R² COMPARISON
        # -----------------------------------------------

        if "R²" in results_df.columns:

            r2_data = results_df[
                ["Model", "R²"]
            ].copy()

            r2_data = r2_data.set_index(
                "Model"
            )

            st.markdown(
                "#### 🎯 R² Score Comparison"
            )

            st.bar_chart(
                r2_data,
                use_container_width=True
            )

        # -----------------------------------------------
        # ERROR METRICS
        # -----------------------------------------------

        error_metrics = [
            "MAE",
            "RMSE"
        ]

        available_errors = [
            metric
            for metric in error_metrics
            if metric in results_df.columns
        ]

        if available_errors:

            error_data = results_df[
                ["Model"] + available_errors
            ].copy()

            error_data = error_data.set_index(
                "Model"
            )

            st.markdown(
                "#### 📉 Error Comparison"
            )

            st.bar_chart(
                error_data,
                use_container_width=True
            )

        # -----------------------------------------------
        # BEST MODEL
        # -----------------------------------------------

        if "R²" in results_df.columns:

            best_row = results_df.iloc[0]

            st.success(
                f"🏆 Best Regression Model: "
                f"**{best_row['Model']}** "
                f"(R²: {best_row['R²']:.4f})"
            )

def run_classification(data, target):

    work_df = data.copy()

    X = work_df.drop(
        columns=[target]
    )

    y = work_df[target]

    # --------------------------------------------------------
    # IDENTIFY FEATURES
    # --------------------------------------------------------

    numeric_features = (
        X.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude=np.number
        ).columns.tolist()
    )

    if not numeric_features and not categorical_features:

        raise ValueError(
            "No usable feature columns found."
        )

    # --------------------------------------------------------
    # REMOVE MISSING TARGET ROWS
    # --------------------------------------------------------

    valid_target = y.notna()

    X = X.loc[
        valid_target
    ]

    y = y.loc[
        valid_target
    ]

    # --------------------------------------------------------
    # VALIDATE TARGET
    # --------------------------------------------------------

    if y.nunique() < 2:

        raise ValueError(
            "Classification target must contain "
            "at least two classes."
        )

    if y.nunique() > 20:

        raise ValueError(
            "Target contains too many classes "
            "for automatic classification."
        )

    # --------------------------------------------------------
    # ENCODE TARGET
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y.astype(str)
    )

    # --------------------------------------------------------
    # NUMERIC PIPELINE
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # COLUMN TRANSFORMER
    # --------------------------------------------------------

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            )
    }

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        try:

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            # -----------------------------------------------
            # METRICS
            # -----------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({

                "Model": name,

                "Accuracy": round(
                    accuracy,
                    4
                ),

                "Precision": round(
                    precision,
                    4
                ),

                "Recall": round(
                    recall,
                    4
                ),

                "F1 Score": round(
                    f1,
                    4
                )
            })

            trained_models[
                name
            ] = pipeline

        except Exception:

            continue

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        raise ValueError(
            "Unable to train any classification model."
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SORT BY F1 SCORE
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "F1 Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0][
        "Model"
    ]

    best_pipeline = trained_models[
        best_model_name
    ]

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance_df = None

    try:

        trained_preprocessor = (
            best_pipeline.named_steps[
                "preprocessor"
            ]
        )

        trained_model = (
            best_pipeline.named_steps[
                "model"
            ]
        )

        feature_names = (
            trained_preprocessor
            .get_feature_names_out()
        )

        # Tree-based models
        if hasattr(
            trained_model,
            "feature_importances_"
        ):

            importances = (
                trained_model
                .feature_importances_
            )

        # Logistic Regression
        elif hasattr(
            trained_model,
            "coef_"
        ):

            importances = np.mean(
                np.abs(
                    trained_model.coef_
                ),
                axis=0
            )

        else:

            importances = None

        if importances is not None:

            feature_importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    importances

            })

            feature_importance_df = (
                feature_importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

    except Exception:

        feature_importance_df = None

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (
        results_df,
        best_model_name,
        best_pipeline,
        feature_importance_df
    )
# ============================================================
# REGRESSION MODEL COMPARISON + FEATURE IMPORTANCE
# ============================================================

def run_regression(data, target):

    work_df = data.copy()

    # --------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X = work_df.drop(
        columns=[target]
    )

    y = work_df[target]

    # Regression uses numerical features
    X = X.select_dtypes(
        include=np.number
    )

    if X.empty:

        raise ValueError(
            "No numerical feature columns found "
            "for regression."
        )

    # --------------------------------------------------------
    # REMOVE MISSING TARGET VALUES
    # --------------------------------------------------------

    valid_rows = y.notna()

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]

    # --------------------------------------------------------
    # HANDLE INFINITE VALUES
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # HANDLE MISSING FEATURES
    # --------------------------------------------------------

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # CLEAN TARGET
    # --------------------------------------------------------

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    y = y.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_target = y.notna()

    X = X.loc[
        valid_target
    ]

    y = y.loc[
        valid_target
    ]

    # --------------------------------------------------------
    # MINIMUM DATA CHECK
    # --------------------------------------------------------

    if len(X) < 20:

        raise ValueError(
            "At least 20 valid rows are recommended "
            "for regression."
        )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )
    }

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # TRAIN AND EVALUATE MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        try:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            # -----------------------------------------------
            # MAE
            # -----------------------------------------------

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            # -----------------------------------------------
            # RMSE
            # -----------------------------------------------

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            # -----------------------------------------------
            # R² SCORE
            # -----------------------------------------------

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({

                "Model": name,

                "MAE": round(
                    mae,
                    4
                ),

                "RMSE": round(
                    rmse,
                    4
                ),

                "R²": round(
                    r2,
                    4
                )
            })

            trained_models[
                name
            ] = model

        except Exception:

            continue

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        raise ValueError(
            "Unable to train any regression model."
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # SORT BY R²
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        "R²",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0][
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance_df = None

    try:

        if hasattr(
            best_model,
            "feature_importances_"
        ):

            feature_importance_df = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    best_model.feature_importances_

            })

        elif hasattr(
            best_model,
            "coef_"
        ):

            feature_importance_df = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    np.abs(
                        best_model.coef_
                    )

            })

        if feature_importance_df is not None:

            feature_importance_df = (
                feature_importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

    except Exception:

        feature_importance_df = None

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return (
        results_df,
        best_model_name,
        best_model,
        feature_importance_df
    )

# ============================================================
# SMART SYNTHETIC DATASET GENERATOR
# ============================================================

# ============================================================
# SMART SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_dataset(
    dataset_type,
    rows,
    features,
    anomaly_percentage=5,
    missing_percentage=0,
    duplicate_percentage=0,
    outlier_percentage=0,
    noise_level=0.1
):

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # CLASSIFICATION DATASET
    # --------------------------------------------------------

    if dataset_type == "Classification":

        X, y = make_classification(
            n_samples=rows,
            n_features=features,
            n_informative=max(
                2,
                min(features, int(features * 0.6))
            ),
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            class_sep=1.2,
            random_state=42
        )

        columns = [
            f"Feature_{i+1}"
            for i in range(features)
        ]

        generated_df = pd.DataFrame(
            X,
            columns=columns
        )

        generated_df["Target"] = y


    # --------------------------------------------------------
    # REGRESSION DATASET
    # --------------------------------------------------------

    elif dataset_type == "Regression":

        X, y = make_regression(
            n_samples=rows,
            n_features=features,
            n_informative=max(
                2,
                min(features, int(features * 0.7))
            ),
            noise=noise_level * 100,
            random_state=42
        )

        columns = [
            f"Feature_{i+1}"
            for i in range(features)
        ]

        generated_df = pd.DataFrame(
            X,
            columns=columns
        )

        generated_df["Target"] = y


    # --------------------------------------------------------
    # ANOMALY DETECTION DATASET
    # --------------------------------------------------------

    elif dataset_type == "Anomaly Detection":

        X = rng.normal(
            loc=0,
            scale=1,
            size=(rows, features)
        )

        columns = [
            f"Feature_{i+1}"
            for i in range(features)
        ]

        generated_df = pd.DataFrame(
            X,
            columns=columns
        )

        # Number of anomalies
        anomaly_count = int(
            rows *
            anomaly_percentage /
            100
        )

        if anomaly_count > 0:

            anomaly_indices = rng.choice(
                rows,
                size=min(
                    anomaly_count,
                    rows
                ),
                replace=False
            )

            # Make selected rows extreme
            generated_df.loc[
                anomaly_indices,
                columns
            ] *= 6


    # --------------------------------------------------------
    # MIXED DATASET
    # --------------------------------------------------------

    elif dataset_type == "Mixed Dataset":

        numeric_count = max(
            2,
            features - 2
        )

        data = {}

        # Numeric columns
        for i in range(numeric_count):

            data[
                f"Numeric_{i+1}"
            ] = rng.normal(
                5000,
                500,
                rows
            )

        # Category column
        data["Category"] = rng.choice(
            [
                "A",
                "B",
                "C",
                "D"
            ],
            rows
        )

        # Status column
        data["Status"] = rng.choice(
            [
                "Active",
                "Inactive"
            ],
            rows
        )

        generated_df = pd.DataFrame(
            data
        )


    else:

        raise ValueError(
            "Invalid dataset type selected."
        )


    # ========================================================
    # ADD NOISE
    # ========================================================

    numeric_columns = generated_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Do not modify Target
    feature_numeric_columns = [
        col
        for col in numeric_columns
        if col != "Target"
    ]

    if (
        noise_level > 0
        and feature_numeric_columns
    ):

        noise = rng.normal(
            0,
            noise_level,
            size=(
                len(generated_df),
                len(feature_numeric_columns)
            )
        )

        generated_df[
            feature_numeric_columns
        ] += noise


    # ========================================================
    # ADD OUTLIERS
    # ========================================================

    if (
        outlier_percentage > 0
        and feature_numeric_columns
    ):

        outlier_count = int(
            len(generated_df) *
            outlier_percentage /
            100
        )

        if outlier_count > 0:

            outlier_indices = rng.choice(
                len(generated_df),
                size=min(
                    outlier_count,
                    len(generated_df)
                ),
                replace=False
            )

            for column in feature_numeric_columns:

                generated_df.loc[
                    outlier_indices,
                    column
                ] *= 8


    # ========================================================
    # ADD MISSING VALUES
    # ========================================================

    if missing_percentage > 0:

        total_cells = (
            len(generated_df) *
            len(generated_df.columns)
        )

        missing_count = int(
            total_cells *
            missing_percentage /
            100
        )

        if missing_count > 0:

            for _ in range(missing_count):

                row_index = rng.integers(
                    0,
                    len(generated_df)
                )

                column_index = rng.integers(
                    0,
                    len(generated_df.columns)
                )

                generated_df.iat[
                    row_index,
                    column_index
                ] = np.nan


    # ========================================================
    # ADD DUPLICATES
    # ========================================================

    if duplicate_percentage > 0:

        duplicate_count = int(
            len(generated_df) *
            duplicate_percentage /
            100
        )

        if duplicate_count > 0:

            duplicate_rows = generated_df.sample(
                n=min(
                    duplicate_count,
                    len(generated_df)
                ),
                random_state=42
            )

            generated_df = pd.concat(
                [
                    generated_df,
                    duplicate_rows
                ],
                ignore_index=True
            )


    # ========================================================
    # SHUFFLE DATASET
    # ========================================================

    generated_df = generated_df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    return generated_df
# ============================================================
# ADVANCED DATASET WORKFLOW
# ============================================================

if df is not None and page == "🚀 Dataset Workflow":

    st.header(
        "🚀 Intelligent Dataset Workflow"
    )

    st.write(
        "DataPulse automatically analyzes your dataset "
        "and creates a processing pipeline based on "
        "the objectives you select."
    )

    # ========================================================
    # AUTOMATIC DATASET SCAN
    # ========================================================

    scan = dataset_scan(
        df
    )

    st.subheader(
        "🔍 Automatic Dataset Scan"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        scan["rows"]
    )

    col2.metric(
        "Columns",
        scan["columns"]
    )

    col3.metric(
        "Numeric Features",
        scan["numeric"]
    )

    col4.metric(
        "Categorical Features",
        scan["categorical"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Missing Values",
        scan["missing"]
    )

    col2.metric(
        "Duplicate Rows",
        scan["duplicates"]
    )

    col3.metric(
        "Infinite Values",
        scan["infinite"]
    )

    col4.metric(
        "Constant Columns",
        scan["constant"]
    )

    # ========================================================
    # SMART RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "🤖 DataPulse Recommendations"
    )

    recommendations = []

    if scan["missing"] > 0:

        recommendations.append(
            "🧹 Data Cleaning is recommended "
            "because missing values were detected."
        )

    if scan["duplicates"] > 0:

        recommendations.append(
            "🔄 Duplicate removal is recommended."
        )

    if scan["numeric"] >= 2:

        recommendations.append(
            "📈 Visualization is recommended "
            "because multiple numerical features exist."
        )

    if scan["numeric"] >= 2:

        recommendations.append(
            "🚨 Anomaly Detection is available."
        )

    classification_target = (
        find_classification_target(df)
    )

    regression_target = (
        find_regression_target(df)
    )

    if classification_target:

        recommendations.append(
            f"🎯 Classification appears suitable. "
            f"Possible target: `{classification_target}`"
        )

    if regression_target:

        recommendations.append(
            f"📉 Regression appears suitable. "
            f"Possible target: `{regression_target}`"
        )

    if not recommendations:

        recommendations.append(
            "ℹ️ No special processing recommendations "
            "were detected."
        )

    for recommendation in recommendations:

        st.info(
            recommendation
        )

    st.divider()

    # ========================================================
    # OBJECTIVE SELECTION
    # ========================================================

    st.subheader(
        "🎯 Select Your Objectives"
    )

    st.write(
        "Choose what you want DataPulse to perform "
        "on this dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        profiling_check = st.checkbox(
            "📊 Analyze Dataset Quality",
            value=True
        )

        cleaning_check = st.checkbox(
            "🧹 Clean & Prepare Dataset"
        )

        visualization_check = st.checkbox(
            "📈 Generate Visual Analytics"
        )

        anomaly_check = st.checkbox(
            "🚨 Detect Anomalies"
        )

    with col2:

        classification_check = st.checkbox(
            "🎯 Build Classification Models"
        )

        regression_check = st.checkbox(
            "📉 Build Regression Models"
        )

        insights_check = st.checkbox(
            "🤖 Generate AI Insights"
        )

    # ========================================================
    # ANOMALY CONFIGURATION
    # ========================================================

    anomaly_models = []

    if anomaly_check:

        st.subheader(
            "🚨 Anomaly Detection Configuration"
        )

        anomaly_models = st.multiselect(
            "Select anomaly algorithms",
            [
                "Isolation Forest",
                "DBSCAN",
                "One-Class SVM"
            ],
            default=[
                "Isolation Forest",
                "DBSCAN",
                "One-Class SVM"
            ],
            key="workflow_anomaly_models"
        )

        if not anomaly_models:

            st.warning(
                "Select at least one anomaly algorithm."
            )

    # ========================================================
    # CLASSIFICATION CONFIGURATION
    # ========================================================

    classification_target_selected = None

    if classification_check:

        st.subheader(
            "🎯 Classification Configuration"
        )

        classification_target_selected = st.selectbox(
            "Select classification target",
            df.columns.tolist(),
            index=(
                df.columns.tolist().index(
                    classification_target
                )
                if classification_target
                in df.columns
                else 0
            ),
            key="workflow_classification_target"
        )

        st.caption(
            "DataPulse will automatically compare "
            "multiple classification models."
        )

    # ========================================================
    # REGRESSION CONFIGURATION
    # ========================================================

    regression_target_selected = None

    if regression_check:

        st.subheader(
            "📉 Regression Configuration"
        )

        regression_columns = (
            get_numeric_columns(
                df
            )
        )

        if regression_columns:

            regression_target_selected = st.selectbox(
                "Select regression target",
                regression_columns,
                index=(
                    regression_columns.index(
                        regression_target
                    )
                    if regression_target
                    in regression_columns
                    else 0
                ),
                key="workflow_regression_target"
            )

        else:

            st.error(
                "❌ No numerical target is available "
                "for regression."
            )

    st.divider()

    # ========================================================
    # GENERATE PIPELINE
    # ========================================================

    if st.button(
        "🚀 Generate Results",
        type="primary",
        use_container_width=True
    ):

        selected_count = sum([
            profiling_check,
            cleaning_check,
            visualization_check,
            anomaly_check,
            classification_check,
            regression_check,
            insights_check
        ])

        if selected_count == 0:

            st.warning(
                "⚠️ Select at least one objective."
            )

            st.stop()

        st.success(
            "🚀 DataPulse pipeline started."
        )

        # ====================================================
        # DATA PROFILING
        # ====================================================

        profiling_results = None

        if profiling_check:

            st.header(
                "📊 Dataset Quality Analysis"
            )

            try:

                quality_score = (
                    calculate_quality_score(
                        df
                    )
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Quality Score",
                    f"{quality_score}/100"
                )

                col2.metric(
                    "Missing Values",
                    scan["missing"]
                )

                col3.metric(
                    "Duplicates",
                    scan["duplicates"]
                )

                col4.metric(
                    "Constant Columns",
                    scan["constant"]
                )

                st.progress(
                    min(
                        max(
                            int(
                                quality_score
                            ),
                            0
                        ),
                        100
                    )
                )

                st.subheader(
                    "📋 Statistical Profile"
                )

                st.dataframe(
                    df.describe(
                        include="all"
                    ).transpose(),
                    use_container_width=True
                )

                st.subheader(
                    "🔎 Validation Results"
                )

                validation_results = (
                    run_validation(
                        df
                    )
                )

                for result in validation_results:

                    st.write(
                        result
                    )

                profiling_results = {
                    "quality_score":
                        quality_score
                }

            except Exception as e:

                st.error(
                    f"❌ Profiling failed: {e}"
                )

        # ====================================================
        # DATA CLEANING
        # ====================================================

        processed_df = df.copy()

        if cleaning_check:

            st.header(
                "🧹 Automated Data Preparation"
            )

            try:

                before_rows = len(
                    processed_df
                )

                before_missing = int(
                    processed_df
                    .isnull()
                    .sum()
                    .sum()
                )

                processed_df = clean_dataset(
                    processed_df
                )

                after_rows = len(
                    processed_df
                )

                after_missing = int(
                    processed_df
                    .isnull()
                    .sum()
                    .sum()
                )

                col1, col2, col3 = (
                    st.columns(3)
                )

                col1.metric(
                    "Rows Before",
                    before_rows
                )

                col2.metric(
                    "Rows After",
                    after_rows
                )

                col3.metric(
                    "Missing Values After",
                    after_missing
                )

                st.success(
                    "✅ Automatic data preparation completed."
                )

                st.dataframe(
                    processed_df.head(50),
                    use_container_width=True
                )

                cleaned_csv = (
                    processed_df
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    "⬇️ Download Prepared Dataset",
                    cleaned_csv,
                    "DataPulse_prepared_dataset.csv",
                    "text/csv",
                    key="workflow_prepared_download"
                )

            except Exception as e:

                st.error(
                    f"❌ Data cleaning failed: {e}"
                )

        # ====================================================
        # VISUAL ANALYTICS
        # ====================================================

        if visualization_check:

            st.header(
                "📈 Automated Visual Analytics"
            )

            try:

                visual_numeric = (
                    get_numeric_columns(
                        processed_df
                    )
                )

                if visual_numeric:

                    selected_visual = st.selectbox(
                        "Select feature for distribution analysis",
                        visual_numeric,
                        key="workflow_visual_feature"
                    )

                    histogram = create_histogram(
                        processed_df,
                        selected_visual
                    )

                    st.plotly_chart(
                        histogram,
                        use_container_width=True
                    )

                    boxplot = create_boxplot(
                        processed_df,
                        selected_visual
                    )

                    st.plotly_chart(
                        boxplot,
                        use_container_width=True
                    )

                if len(visual_numeric) >= 2:

                    correlation_matrix = (
                        processed_df[
                            visual_numeric
                        ].corr()
                    )

                    heatmap = create_heatmap(
                        correlation_matrix
                    )

                    st.plotly_chart(
                        heatmap,
                        use_container_width=True
                    )

                if len(visual_numeric) >= 2:

                    scatter_x = st.selectbox(
                        "X-axis",
                        visual_numeric,
                        key="workflow_scatter_x"
                    )

                    scatter_y = st.selectbox(
                        "Y-axis",
                        visual_numeric,
                        key="workflow_scatter_y"
                    )

                    scatter = create_scatter(
                        processed_df,
                        scatter_x,
                        scatter_y
                    )

                    st.plotly_chart(
                        scatter,
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"❌ Visualization failed: {e}"
                )

        # ====================================================
        # ANOMALY DETECTION
        # ====================================================

        anomaly_results = {}

        if anomaly_check:

            st.header(
                "🚨 Multi-Model Anomaly Detection"
            )

            for selected_model in anomaly_models:

                try:

                    if selected_model == "Isolation Forest":

                        result = detect_anomalies(
                            processed_df
                        )

                    elif selected_model == "DBSCAN":

                        result = detect_anomalies_dbscan(
                            processed_df
                        )

                    else:

                        result = detect_anomalies_svm(
                            processed_df
                        )

                    anomaly_column = (
                        get_anomaly_column(
                            result
                        )
                    )

                    if anomaly_column is None:

                        raise ValueError(
                            f"{selected_model} did not "
                            "return an anomaly column."
                        )

                    anomaly_count = int(
                        (
                            result[
                                anomaly_column
                            ] == -1
                        ).sum()
                    )

                    anomaly_results[
                        selected_model
                    ] = {
                        "data": result,
                        "count": anomaly_count
                    }

                    st.success(
                        f"✅ {selected_model} completed."
                    )

                except Exception as e:

                    st.error(
                        f"❌ {selected_model} failed: {e}"
                    )

            if anomaly_results:

                comparison_rows = []

                for model_name, model_result in (
                    anomaly_results.items()
                ):

                    count = model_result[
                        "count"
                    ]

                    rate = (
                        count
                        / len(processed_df)
                        * 100
                    )

                    comparison_rows.append({
                        "Model": model_name,
                        "Anomalies": count,
                        "Anomaly Rate (%)":
                            round(
                                rate,
                                2
                            )
                    })

                comparison_df = pd.DataFrame(
                    comparison_rows
                )

                st.subheader(
                    "📊 Anomaly Model Comparison"
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                anomaly_chart = px.bar(
                    comparison_df,
                    x="Model",
                    y="Anomalies",
                    title="Anomalies Detected by Model"
                )

                st.plotly_chart(
                    anomaly_chart,
                    use_container_width=True
                )

                # Show best agreement candidate
                st.info(
                    "💡 Models can identify different "
                    "types of unusual records. "
                    "Comparing their results gives a "
                    "more reliable view of dataset anomalies."
                )

                # Display first selected model results
                first_model = list(
                    anomaly_results.keys()
                )[0]

                first_result = anomaly_results[
                    first_model
                ]["data"]

                first_anomaly_column = (
                    get_anomaly_column(
                        first_result
                    )
                )

                st.subheader(
                    f"⚠️ {first_model} Anomalies"
                )

                anomaly_rows = first_result[
                    first_result[
                        first_anomaly_column
                    ] == -1
                ]

                st.dataframe(
                    anomaly_rows.head(50),
                    use_container_width=True
                )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        classification_results = None

        if classification_check:

            st.header(
                "🎯 Automated Classification"
            )

            try:

                (
        classification_results,
        best_classification_model,
        best_classifier,
        classification_feature_importance
    ) = run_classification(
        processed_df,
        classification_target_selected
    )

                st.subheader(
                    "📊 Model Comparison"
                )

                st.dataframe(
                    classification_results,
                    use_container_width=True
                )

                st.success(
                    f"🏆 Best Classification Model: "
                    f"{best_classification_model}"
                )
                
# ============================================================
# CLASSIFICATION MODEL COMPARISON CHART
# ============================================================

                show_model_comparison_charts(
    classification_results,
    "Classification"
)

                # ============================================================
                # CLASSIFICATION FEATURE IMPORTANCE
                # ============================================================

                if classification_feature_importance is not None:

                    st.subheader(
        "🔍 Feature Importance"
    )

                    st.caption(
        f"Features influencing the "
        f"{best_classification_model} model"
    )

                    top_features = (
        classification_feature_importance
        .head(10)
        .copy()
    )

                    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

                    fig = px.bar(
        top_features.set_index(
            "Feature"
        )["Importance"]
    )

                fig = px.bar(
                    classification_results,
                    x="Model",
                    y="F1 Score",
                    title="Classification Model Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"❌ Classification failed: {e}"
                )

        # ====================================================
        # REGRESSION
        # ====================================================

        regression_results = None

        if regression_check:

            st.header(
                "📉 Automated Regression"
            )

            try:

                (
        regression_results,
        best_regression_model,
        best_regressor,
        regression_feature_importance
    ) = run_regression(
        processed_df,
        regression_target_selected
    )

                st.subheader(
                    "📊 Model Comparison"
                )

                st.dataframe(
                    regression_results,
                    use_container_width=True
                )

                st.success(
                    f"🏆 Best Regression Model: "
                    f"{best_regression_model}"
                )
                # ============================================================
                # REGRESSION MODEL COMPARISON CHART
                # ============================================================

                show_model_comparison_charts(
    regression_results,
    "Regression"
)
                # ============================================================
                # REGRESSION FEATURE IMPORTANCE
                # ============================================================

                if regression_feature_importance is not None:

                    st.subheader(
        "🔍 Feature Importance"
    )

                    st.caption(
        f"Features influencing the "
        f"{best_regression_model} model"
    )

                    top_features = (
        regression_feature_importance
        .head(10)
        .copy()
    )

                    st.dataframe(
        top_features,
        use_container_width=True,
        hide_index=True
    )

                    st.bar_chart(
        top_features.set_index(
            "Feature"
        )["Importance"]
    )

                fig = px.bar(
                    regression_results,
                    x="Model",
                    y="R²",
                    title="Regression Model Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"❌ Regression failed: {e}"
                )

        # ====================================================
        # AI INSIGHTS
        # ====================================================

        if insights_check:

            st.header(
                "🤖 DataPulse AI Insights"
            )

            try:

                if anomaly_results:

                    first_model = list(
                        anomaly_results.keys()
                    )[0]

                    insight_data = (
                        anomaly_results[
                            first_model
                        ]["data"]
                    )

                    insight_anomaly_count = (
                        anomaly_results[
                            first_model
                        ]["count"]
                    )

                else:

                    insight_data = processed_df

                    insight_anomaly_count = 0

                insights = generate_insights(
                    insight_data,
                    insight_anomaly_count
                )

                for insight in insights:

                    st.success(
                        insight
                    )

            except Exception as e:

                st.warning(
                    f"⚠️ AI Insights unavailable: {e}"
                )

        # ====================================================
        # FINAL PIPELINE SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "📋 Pipeline Summary"
        )

        summary_items = []

        if profiling_check:
            summary_items.append(
                "📊 Dataset Profiling"
            )

        if cleaning_check:
            summary_items.append(
                "🧹 Data Preparation"
            )

        if visualization_check:
            summary_items.append(
                "📈 Visual Analytics"
            )

        if anomaly_check:
            summary_items.append(
                "🚨 Anomaly Detection"
            )

        if classification_check:
            summary_items.append(
                "🎯 Classification"
            )

        if regression_check:
            summary_items.append(
                "📉 Regression"
            )

        if insights_check:
            summary_items.append(
                "🤖 AI Insights"
            )

        for item in summary_items:

            st.write(
                f"✅ {item}"
            )

        st.success(
            "🎉 DataPulse completed the selected "
            "dataset-analysis pipeline."
        )


# ============================================================
# DASHBOARD PAGE
# ============================================================

if df is not None and page == "Dashboard":

    st.header("📌 Key Metrics")

    total_rows = df.shape[0]
    total_columns = df.shape[1]

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        total_rows
    )

    col2.metric(
        "Total Columns",
        total_columns
    )

    col3.metric(
        "Missing Values",
        missing_values
    )

    col4.metric(
        "Duplicate Rows",
        duplicate_rows
    )

    st.header(
        "📈 Data Quality Score"
    )

    quality_score = calculate_quality_score(
        df
    )

    st.progress(
        min(
            max(
                int(
                    quality_score
                ),
                0
            ),
            100
        )
    )

    st.metric(
        "Dataset Health Score",
        f"{quality_score}/100"
    )

    if quality_score >= 90:

        st.success(
            "Excellent Dataset Quality"
        )

    elif quality_score >= 70:

        st.warning(
            "Moderate Dataset Quality"
        )

    else:

        st.error(
            "Poor Dataset Quality"
        )

    st.header(
        "📂 Dataset Overview"
    )

    st.write(
        f"Dataset Shape: {df.shape}"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.header(
        "📉 Missing Values"
    )

    missing_series = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column":
            missing_series.index,
        "Missing Values":
            missing_series.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.header(
        "📊 Statistical Summary"
    )

    st.dataframe(
        df.describe(
            include="all"
        ),
        use_container_width=True
    )

    st.header(
        "✅ Validation Results"
    )

    validation_results = run_validation(
        df
    )

    for result in validation_results:

        st.write(result)


# ============================================================
# ANOMALY DETECTION PAGE
# ============================================================

if df is not None and page == "Anomaly Detection":

    st.header(
        "🚨 Anomaly Detection"
    )

    model_choice = st.selectbox(
        "Select Detection Model",
        [
            "Isolation Forest",
            "DBSCAN",
            "One-Class SVM"
        ]
    )

    try:

        if model_choice == "Isolation Forest":

            result_df = detect_anomalies(
                df
            )

        elif model_choice == "DBSCAN":

            result_df = detect_anomalies_dbscan(
                df
            )

        else:

            result_df = detect_anomalies_svm(
                df
            )

        anomaly_column = get_anomaly_column(
            result_df
        )

        if anomaly_column is None:

            raise ValueError(
                "Model did not return an anomaly column."
            )

        st.success(
            f"✅ {model_choice} completed successfully!"
        )

        anomaly_count = int(
            (
                result_df[
                    anomaly_column
                ] == -1
            ).sum()
        )

        normal_count = (
            len(result_df)
            - anomaly_count
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            len(result_df)
        )

        col2.metric(
            "Normal Records",
            normal_count
        )

        col3.metric(
            "Anomalies",
            anomaly_count
        )

        anomaly_df = result_df[
            result_df[
                anomaly_column
            ] == -1
        ]

        st.subheader(
            "⚠️ Detected Anomalies"
        )

        st.dataframe(
            anomaly_df.head(50),
            use_container_width=True
        )

        numeric_columns = get_numeric_columns(
            df
        )

        if len(numeric_columns) >= 2:

            x_axis = st.selectbox(
                "Select X-axis",
                numeric_columns,
                key="anomaly_x_axis"
            )

            y_axis = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                key="anomaly_y_axis"
            )

            fig = px.scatter(
                result_df,
                x=x_axis,
                y=y_axis,
                color=result_df[
                    anomaly_column
                ].astype(str),
                title="Anomaly Visualization"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.header(
            "🤖 AI Insights"
        )

        insights = generate_insights(
            result_df,
            anomaly_count
        )

        for insight in insights:

            st.success(
                insight
            )

        csv_data = (
            anomaly_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Anomaly Report",
            csv_data,
            "DataPulse_anomaly_report.csv",
            "text/csv"
        )

        quality_score = calculate_quality_score(
            df
        )

        try:

            save_analysis(
                uploaded_file.name,
                quality_score,
                anomaly_count,
                model_choice
            )

        except Exception:
            pass

    except ValueError as e:

        st.error(
            f"⚠️ {model_choice} cannot process this dataset."
        )

        st.warning(
            str(e)
        )

    except Exception as e:

        st.error(
            f"❌ An error occurred while running "
            f"{model_choice}."
        )

        st.code(
            str(e)
        )


# ============================================================
# DATA CLEANING PAGE
# ============================================================

if df is not None and page == "Data Cleaning":

    st.header(
        "🧹 Data Cleaning & Resolution"
    )

    st.subheader(
        "📊 Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(
            df.isnull().sum().sum()
        )
    )

    col4.metric(
        "Duplicate Rows",
        int(
            df.duplicated().sum()
        )
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    st.subheader(
        "⚠️ Missing Values"
    )

    missing_df = (
        df.isnull()
        .sum()
    )

    missing_df = missing_df[
        missing_df > 0
    ].sort_values(
        ascending=False
    )

    if missing_df.empty:

        st.success(
            "✅ No missing values found!"
        )

    else:

        st.dataframe(
            pd.DataFrame({
                "Column":
                    missing_df.index,
                "Missing Values":
                    missing_df.values
            }),
            use_container_width=True
        )

        numeric_columns = get_numeric_columns(
            df
        )

        categorical_columns = get_categorical_columns(
            df
        )

        cleaning_method = st.selectbox(
            "Choose a missing-value method",
            [
                "Fill numerical values with Median",
                "Fill numerical values with Mean",
                "Fill categorical values with Mode",
                "Remove rows containing missing values"
            ]
        )

        if st.button(
            "🧹 Clean Missing Values",
            key="clean_missing"
        ):

            cleaned_df = df.copy()

            if cleaning_method == (
                "Fill numerical values with Median"
            ):

                for column in numeric_columns:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(
                            cleaned_df[
                                column
                            ].median()
                        )
                    )

            elif cleaning_method == (
                "Fill numerical values with Mean"
            ):

                for column in numeric_columns:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(
                            cleaned_df[
                                column
                            ].mean()
                        )
                    )

            elif cleaning_method == (
                "Fill categorical values with Mode"
            ):

                for column in categorical_columns:

                    mode = (
                        cleaned_df[
                            column
                        ].mode()
                    )

                    if not mode.empty:

                        cleaned_df[column] = (
                            cleaned_df[column]
                            .fillna(
                                mode.iloc[0]
                            )
                        )

            else:

                cleaned_df = (
                    cleaned_df
                    .dropna()
                    .reset_index(drop=True)
                )

            st.success(
                "✅ Missing values cleaned."
            )

            st.dataframe(
                cleaned_df.head(50),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Cleaned Dataset",
                cleaned_df.to_csv(
                    index=False
                ),
                "DataPulse_cleaned.csv",
                "text/csv",
                key="clean_missing_download"
            )

    # ========================================================
    # DUPLICATES
    # ========================================================

    st.subheader(
        "🔄 Duplicate Rows"
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows Found",
        duplicate_count
    )

    if duplicate_count == 0:

        st.success(
            "✅ No duplicate rows found!"
        )

    else:

        duplicate_rows = df[
            df.duplicated(
                keep=False
            )
        ]

        st.dataframe(
            duplicate_rows.head(50),
            use_container_width=True
        )

        if st.button(
            "🧹 Remove Duplicate Rows",
            key="remove_duplicates"
        ):

            cleaned_duplicates = (
                df
                .drop_duplicates()
                .reset_index(drop=True)
            )

            st.success(
                "✅ Duplicate rows removed."
            )

            st.dataframe(
                cleaned_duplicates.head(50),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Dataset Without Duplicates",
                cleaned_duplicates.to_csv(
                    index=False
                ),
                "DataPulse_no_duplicates.csv",
                "text/csv",
                key="duplicates_download"
            )

    # ========================================================
    # OUTLIERS
    # ========================================================

    st.subheader(
        "📈 Outlier Detection"
    )

    numeric_columns = get_numeric_columns(
        df
    )

    if numeric_columns:

        outlier_column = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="manual_outlier_column"
        )

        q1 = df[
            outlier_column
        ].quantile(0.25)

        q3 = df[
            outlier_column
        ].quantile(0.75)

        iqr = q3 - q1

        lower_limit = q1 - (
            1.5 * iqr
        )

        upper_limit = q3 + (
            1.5 * iqr
        )

        outlier_mask = (
            (
                df[
                    outlier_column
                ] < lower_limit
            )
            |
            (
                df[
                    outlier_column
                ] > upper_limit
            )
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        st.metric(
            "Outliers Detected",
            outlier_count
        )

        if outlier_count > 0:

            st.dataframe(
                df[outlier_mask].head(50),
                use_container_width=True
            )

    else:

        st.info(
            "No numerical columns available."
        )


# ============================================================
# VISUALIZATION PAGE
# ============================================================

if df is not None and page == "Visualizations":

    st.header(
        "📈 Data Visualizations"
    )

    numeric_columns = get_numeric_columns(
        df
    )

    if not numeric_columns:

        st.warning(
            "⚠️ No numerical columns available."
        )

    else:

        selected_column = st.selectbox(
            "Select Column",
            numeric_columns,
            key="visual_histogram"
        )

        hist_fig = create_histogram(
            df,
            selected_column
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

        if len(numeric_columns) >= 2:

            st.subheader(
                "Correlation Heatmap"
            )

            correlation_matrix = (
                df[
                    numeric_columns
                ].corr()
            )

            heatmap_fig = create_heatmap(
                correlation_matrix
            )

            st.plotly_chart(
                heatmap_fig,
                use_container_width=True
            )

            st.subheader(
                "Box Plot"
            )

            box_column = st.selectbox(
                "Select Column",
                numeric_columns,
                key="visual_boxplot"
            )

            box_fig = create_boxplot(
                df,
                box_column
            )

            st.plotly_chart(
                box_fig,
                use_container_width=True
            )

            st.subheader(
                "Scatter Plot"
            )

            x_feature = st.selectbox(
                "X Feature",
                numeric_columns,
                key="visual_scatter_x"
            )

            y_feature = st.selectbox(
                "Y Feature",
                numeric_columns,
                key="visual_scatter_y"
            )

            scatter_fig = create_scatter(
                df,
                x_feature,
                y_feature
            )

            st.plotly_chart(
                scatter_fig,
                use_container_width=True
            )


# ============================================================
# REAL-TIME MONITORING
# ============================================================

if df is not None and page == "Real-Time Monitoring":

    st.header(
        "📡 Real-Time Monitoring"
    )

    st_autorefresh(
        interval=5000,
        key="realtime_refresh"
    )

    sample_size = min(
        200,
        len(df)
    )

    live_data = df.sample(
        sample_size,
        replace=False
    ).copy()

    numeric_columns = get_numeric_columns(
        live_data
    )

    for column in numeric_columns:

        live_data[column] = (
            live_data[column]
            + np.random.normal(
                0,
                0.5,
                size=len(live_data)
            )
        )

    try:

        result_df = detect_anomalies(
            live_data
        )

        anomaly_column = get_anomaly_column(
            result_df
        )

        if anomaly_column:

            anomaly_count = int(
                (
                    result_df[
                        anomaly_column
                    ] == -1
                ).sum()
            )

            st.metric(
                "Live Anomalies Detected",
                anomaly_count
            )

        st.subheader(
            "📊 Live Data Stream"
        )

        st.dataframe(
            result_df.head(50),
            use_container_width=True
        )

        if len(numeric_columns) >= 2:

            x_axis = st.selectbox(
                "X-axis",
                numeric_columns,
                key="live_x"
            )

            y_axis = st.selectbox(
                "Y-axis",
                numeric_columns,
                key="live_y"
            )

            realtime_fig = px.scatter(
                result_df,
                x=x_axis,
                y=y_axis,
                color=result_df[
                    anomaly_column
                ].astype(str),
                title="Live Anomaly Monitoring"
            )

            st.plotly_chart(
                realtime_fig,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Real-time monitoring failed: {e}"
        )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

if page == "Analysis History":

    st.header(
        "📜 Analysis History"
    )

    try:

        history_df = load_history()

    except Exception as e:

        st.error(
            f"Unable to load history: {e}"
        )

        history_df = pd.DataFrame()

    if history_df.empty:

        st.warning(
            "No analysis history available."
        )

    else:

        total_runs = len(
            history_df
        )

        avg_quality = round(
            history_df[
                "Quality Score"
            ].mean(),
            2
        )

        total_anomalies = (
            history_df[
                "Anomalies"
            ].sum()
        )

        most_used_model = (
            history_df[
                "Model"
            ].mode()[0]
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Total Analyses",
            total_runs
        )

        col2.metric(
            "Avg Quality Score",
            avg_quality
        )

        col3.metric(
            "Total Anomalies",
            total_anomalies
        )

        col4.metric(
            "Most Used Model",
            most_used_model
        )

        st.subheader(
            "📊 Model Usage"
        )

        model_counts = (
            history_df[
                "Model"
            ]
            .value_counts()
            .reset_index()
        )

        model_counts.columns = [
            "Model",
            "Count"
        ]

        fig = px.bar(
            model_counts,
            x="Model",
            y="Count",
            title="Model Usage Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "📈 Quality Score Trend"
        )

        trend_fig = px.line(
            history_df,
            x="Date",
            y="Quality Score",
            markers=True,
            title="Quality Score Over Time"
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True
        )

        st.subheader(
            "🚨 Anomaly Trend"
        )

        anomaly_fig = px.line(
            history_df,
            x="Date",
            y="Anomalies",
            markers=True,
            title="Anomalies Over Time"
        )

        st.plotly_chart(
            anomaly_fig,
            use_container_width=True
        )

        search_dataset = st.text_input(
            "🔍 Search Dataset"
        )

        filtered_df = history_df

        if search_dataset:

            filtered_df = history_df[
                history_df[
                    "Dataset"
                ].astype(str)
                .str.contains(
                    search_dataset,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True
        )
