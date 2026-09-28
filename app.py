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

from streamlit_autorefresh import (
    st_autorefresh
)

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="DataPulse",
    layout="wide"
)
from streamlit_autorefresh import st_autorefresh
import numpy as np
from profiling.quality_score import calculate_quality_score
from insights.ai_insights import generate_insights
from anomaly_detection.isolation_forest import detect_anomalies

import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="DataPulse",
    layout="wide"
)

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("📊 DataPulse")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Anomaly Detection",
        "Data Cleaning",
        "Visualizations",
        "Real-Time Monitoring",
        "Analysis History"
        
    ]
)

# --------------------------------
# TITLE
# --------------------------------

st.title("📊 DataPulse")

st.subheader(
    "AI-Powered Data Quality & Anomaly Detection Platform"
)

# --------------------------------
# FILE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
# ==============================================
# DASHBOARD PAGE
# ==============================================

if uploaded_file and page == "Dashboard":
    # --------------------------------
    # KPI METRICS
    # --------------------------------

    st.header("📌 Key Metrics")

    total_rows = df.shape[0]
    total_columns = df.shape[1]

    missing_values = df.isnull().sum().sum()

    duplicate_rows = df.duplicated().sum()

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

    # --------------------------------
    # DATA QUALITY SCORE
    # --------------------------------

    st.header("📈 Data Quality Score")

    quality_score = calculate_quality_score(df)

    st.progress(int(quality_score))

    st.metric(
        "Dataset Health Score",
        f"{quality_score}/100"
    )

    if quality_score >= 90:
        st.success("Excellent Dataset Quality")

    elif quality_score >= 70:
        st.warning("Moderate Dataset Quality")

    else:
        st.error("Poor Dataset Quality")

    # --------------------------------
    # DATASET OVERVIEW
    # --------------------------------

    st.header("📂 Dataset Overview")

    st.write("Shape of Dataset:")
    st.write(df.shape)

    st.write("First 5 Rows:")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # --------------------------------
    # MISSING VALUES
    # --------------------------------

    st.header("📉 Missing Values")

    missing_values_series = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values_series.index,
        "Missing Values": missing_values_series.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    # --------------------------------
    # STATISTICAL SUMMARY
    # --------------------------------

    st.header("📊 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )
    st.header("✅ Validation Results")

    validation_results = run_validation(df)

    for result in validation_results:

        st.write(result)

# ==============================================
# ANOMALY DETECTION PAGE
# ==============================================

if uploaded_file and page == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    # Reset uploaded file pointer
    uploaded_file.seek(0)

    # Read uploaded CSV
    try:
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        st.error(
            "❌ The uploaded CSV is empty or could not be read. "
            "Please upload the file again."
        )
        st.stop()

    except Exception as e:
        st.error(
            f"❌ Error reading uploaded CSV: {e}"
        )
        st.stop()

    # Select detection model
    model_choice = st.selectbox(
        "Select Detection Model",
        [
            "Isolation Forest",
            "DBSCAN",
            "One-Class SVM"
        ]
    )

    # Run selected model
    if model_choice == "Isolation Forest":

        result_df = detect_anomalies(df)

    elif model_choice == "DBSCAN":

        result_df = detect_anomalies_dbscan(df)

    elif model_choice == "One-Class SVM":

        result_df = detect_anomalies_svm(df)

    # Display selected model
    st.info(
        f"Current Model: {model_choice}"
    )

    # Count anomalies
    anomaly_count = (
        result_df["anomaly"] == -1
    ).sum()

    # Calculate quality score
    quality_score = calculate_quality_score(df)

    # Save analysis history
    save_analysis(
        uploaded_file.name,
        quality_score,
        anomaly_count,
        model_choice
    )

    # Display anomaly count
    st.metric(
        "Total Anomalies Detected",
        anomaly_count
    )

    # Display anomalous rows
    anomaly_df = result_df[
        result_df["anomaly"] == -1
    ]

    st.subheader("⚠️ Detected Anomalies")

    st.dataframe(
        anomaly_df.head(50),
        use_container_width=True
    )
        # --------------------------------
    # ANOMALY VISUALIZATION
    # --------------------------------

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if len(numeric_columns) >= 2:

        x_axis = st.selectbox(
            "Select X-axis",
            numeric_columns,
            key="x_axis"
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_columns,
            key="y_axis"
        )

        fig = px.scatter(
            result_df,
            x=x_axis,
            y=y_axis,
            color=result_df["anomaly"].astype(str),
            title="Anomaly Visualization"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ At least two numerical columns are required "
            "for anomaly visualization."
        )

    # --------------------------------
    # AI INSIGHTS
    # --------------------------------

    st.header("🤖 AI Insights")

    insights = generate_insights(
        result_df,
        anomaly_count
    )

    for insight in insights:

        st.success(insight)

    # --------------------------------
    # DOWNLOAD REPORTS
    # --------------------------------

    st.header("📥 Download Reports")

    csv_data = anomaly_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Anomaly Report CSV",
        data=csv_data,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )
    
    # ==============================================
# DATA CLEANING PAGE
# ==============================================

if uploaded_file and page == "Data Cleaning":

    st.header("🧹 Data Cleaning & Resolution")

    # Reset uploaded file pointer
    uploaded_file.seek(0)

    # Read uploaded CSV
    try:
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        st.error(
            "❌ The uploaded CSV is empty. "
            "Please upload a valid CSV file."
        )
        st.stop()

    except Exception as e:
        st.error(
            f"❌ Unable to read the CSV file: {e}"
        )
        st.stop()
          
    # ==============================================
    # DATASET OVERVIEW
    # ==============================================

    st.subheader("📊 Dataset Overview")

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
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    # ==============================================
    # MISSING VALUES
    # ==============================================

    st.subheader("⚠️ Missing Values")

    missing_df = df.isnull().sum()

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

        missing_table = pd.DataFrame({
            "Column": missing_df.index,
            "Missing Values": missing_df.values
        })

        st.dataframe(
            missing_table,
            use_container_width=True
        )

        # ==============================================
        # CLEANING METHOD
        # ==============================================

        st.subheader("🔧 Fix Missing Values")

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        cleaning_method = st.selectbox(
            "Choose a method",
            [
                "Fill numerical values with Median",
                "Fill numerical values with Mean",
                "Fill categorical values with Mode",
                "Remove rows containing missing values"
            ]
        )

        # ==============================================
        # CLEAN DATA
        # ==============================================

        if st.button("🧹 Clean Missing Values"):

            cleaned_df = df.copy()

            if cleaning_method == \
                    "Fill numerical values with Median":

                for column in numeric_columns:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(
                            cleaned_df[column].median()
                        )
                    )

            elif cleaning_method == \
                    "Fill numerical values with Mean":

                for column in numeric_columns:

                    cleaned_df[column] = (
                        cleaned_df[column]
                        .fillna(
                            cleaned_df[column].mean()
                        )
                    )

            elif cleaning_method == \
                    "Fill categorical values with Mode":

                for column in categorical_columns:

                    if not cleaned_df[column].mode().empty:

                        cleaned_df[column] = (
                            cleaned_df[column]
                            .fillna(
                                cleaned_df[column].mode()[0]
                            )
                        )

            elif cleaning_method == \
                    "Remove rows containing missing values":

                cleaned_df = cleaned_df.dropna()

            # ==============================================
            # CLEANING RESULT
            # ==============================================

            st.success(
                "✅ Missing values cleaned successfully!"
            )

            remaining_nulls = int(
                cleaned_df.isnull().sum().sum()
            )

            st.metric(
                "Remaining Missing Values",
                remaining_nulls
            )

            st.subheader(
                "📋 Cleaned Dataset Preview"
            )

            st.dataframe(
                cleaned_df.head(50),
                use_container_width=True
            )

            # ==============================================
            # DOWNLOAD CLEAN DATASET
            # ==============================================

            csv = cleaned_df.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Cleaned Dataset",
                csv,
                "DataPulse_cleaned.csv",
                "text/csv"
            )
                # ==============================================
    # DUPLICATE ROW REMOVAL
    # ==============================================

    st.subheader("🔄 Duplicate Rows")

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

        st.warning(
            f"⚠️ {duplicate_count} duplicate rows found."
        )

        st.write("### Duplicate Rows Preview")

        duplicate_rows = df[
            df.duplicated(keep=False)
        ]

        st.dataframe(
            duplicate_rows.head(50),
            use_container_width=True
        )

        if st.button("🧹 Remove Duplicate Rows"):

            cleaned_df = (
                df.drop_duplicates(
                    keep="first"
                )
                .reset_index(drop=True)
            )

            removed_duplicates = (
                len(df) - len(cleaned_df)
            )

            st.success(
                f"✅ Successfully removed "
                f"{removed_duplicates} duplicate rows!"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Rows Before Cleaning",
                len(df)
            )

            col2.metric(
                "Rows After Cleaning",
                len(cleaned_df)
            )

            st.subheader(
                "📋 Cleaned Dataset Preview"
            )

            st.dataframe(
                cleaned_df.head(50),
                use_container_width=True
            )

            cleaned_csv = cleaned_df.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Dataset Without Duplicates",
                cleaned_csv,
                "DataPulse_no_duplicates.csv",
                "text/csv"
            )
      

# ==============================================
# VISUALIZATION PAGE
# ==============================================

if uploaded_file and page == "Visualizations":

    st.header("📈 Data Visualizations")


    numeric_columns = df.select_dtypes(
        include=['float64', 'int64']
    ).columns

    # --------------------------------
    # HISTOGRAM
    # --------------------------------

    if uploaded_file and page == "Visualizations":

        st.header("📈 Data Visualizations")


        numeric_columns = df.select_dtypes(
        include=['float64', 'int64']
        ).columns

    # --------------------------------
    # HISTOGRAM
    # --------------------------------

        st.subheader("Histogram")

        selected_column = st.selectbox(
        "Select Column",
        numeric_columns
        )

        hist_fig = create_histogram(
        df,
        selected_column
        )

        st.plotly_chart(
        hist_fig,
        use_container_width=True
        )

    # --------------------------------
    # CORRELATION HEATMAP
    # --------------------------------

        st.subheader("Correlation Heatmap")

        correlation_matrix = (
        df[numeric_columns]
        .corr()
         )

        heatmap_fig = create_heatmap(
    correlation_matrix
)

        st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )

    # --------------------------------
    # BOXPLOT
    # --------------------------------

    st.subheader("Box Plot")

    box_column = st.selectbox(
        "Select Column for Box Plot",
        numeric_columns,
        key="boxplot"
    )

    box_fig = create_boxplot(
    df,
    box_column
    )

    st.plotly_chart(
    box_fig,
    use_container_width=True
)

    # --------------------------------
    # SCATTER PLOT
    # --------------------------------

    st.subheader("Scatter Plot")

    x_feature = st.selectbox(
        "Select X Feature",
        numeric_columns,
        key="scatter_x"
    )

    y_feature = st.selectbox(
        "Select Y Feature",
        numeric_columns,
        key="scatter_y"
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

# ==============================================
# REAL-TIME MONITORING PAGE
# ==============================================

if uploaded_file and page == "Real-Time Monitoring":

    st.header("📡 Real-Time Monitoring")

    # Auto refresh every 5 seconds
    st_autorefresh(
        interval=5000,
        key="realtime_refresh"
    )


    # Simulate live data
    sample_size = min(200, len(df))

    live_data = df.sample(
    sample_size,
    replace=False
)
    # Numeric columns
    numeric_columns = live_data.select_dtypes(
        include=['float64', 'int64']
    ).columns

    # Add random noise
    for col in numeric_columns:

        live_data[col] = (
            live_data[col]
            + np.random.normal(
                0,
                0.5,
                size=len(live_data)
            )
        )

    # Detect anomalies
    result_df = detect_anomalies(live_data)

    anomaly_count = (
        result_df['anomaly'] == -1
    ).sum()

    # KPI
    st.metric(
        "Live Anomalies Detected",
        anomaly_count
    )

    # Live table
    st.subheader("📊 Live Data Stream")

    st.dataframe(
        result_df.head(50),
        use_container_width=True
    )

    # Scatter plot
    x_axis = st.selectbox(
        "Select X-axis",
        numeric_columns,
        key="live_x"
    )

    y_axis = st.selectbox(
        "Select Y-axis",
        numeric_columns,
        key="live_y"
    )

    realtime_fig = px.scatter(
        result_df,
        x=x_axis,
        y=y_axis,
        color=result_df['anomaly'].astype(str),
        title="Live Anomaly Monitoring"
    )

    st.plotly_chart(
        realtime_fig,
        use_container_width=True
    )
# ==============================================
# ANALYSIS HISTORY PAGE
# ==============================================

if page == "Analysis History":

    st.header("📜 Analysis History")

    history_df = load_history()

    if history_df.empty:

        st.warning(
            "No analysis history available."
        )

    else:

        # KPI Cards
        total_runs = len(history_df)

        avg_quality = round(
            history_df["Quality Score"].mean(),
            2
        )

        total_anomalies = (
            history_df["Anomalies"].sum()
        )

        most_used_model = (
            history_df["Model"].mode()[0]
        )

        col1, col2, col3, col4 = st.columns(4)

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

        # Model Usage Chart
        st.subheader("📊 Model Usage")

        model_counts = (
            history_df["Model"]
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

        # Quality Trend
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

        # Anomaly Trend
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

        # Search
        search_dataset = st.text_input(
            "🔍 Search Dataset"
        )

        filtered_df = history_df

        if search_dataset:

            filtered_df = history_df[
                history_df["Dataset"].str.contains(
                    search_dataset,
                    case=False
                )
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True
        )   