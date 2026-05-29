
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
        "Visualizations",
        "Real-Time Monitoring"
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

# ==============================================
# DASHBOARD PAGE
# ==============================================

if uploaded_file and page == "Dashboard":

    df = pd.read_csv(uploaded_file)

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

# ==============================================
# ANOMALY DETECTION PAGE
# ==============================================

if uploaded_file and page == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    df = pd.read_csv(uploaded_file)

    result_df = detect_anomalies(df)

    anomaly_count = (
        result_df['anomaly'] == -1
    ).sum()

    st.metric(
        "Total Anomalies Detected",
        anomaly_count
    )

    anomaly_df = result_df[
        result_df['anomaly'] == -1
    ]

    st.dataframe(
        anomaly_df.head(50),
        use_container_width=True
    )

    # --------------------------------
    # ANOMALY VISUALIZATION
    # --------------------------------

    numeric_columns = df.select_dtypes(
        include=['float64', 'int64']
    ).columns

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
        color=result_df['anomaly'].astype(str),
        title="Anomaly Visualization"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
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
    ).encode('utf-8')

    st.download_button(
        label="Download Anomaly Report CSV",
        data=csv_data,
        file_name="anomaly_report.csv",
        mime="text/csv"
    )

# ==============================================
# VISUALIZATION PAGE
# ==============================================

if uploaded_file and page == "Visualizations":

    st.header("📈 Data Visualizations")

    df = pd.read_csv(uploaded_file)

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

    hist_fig = px.histogram(
        df,
        x=selected_column,
        title=f"Distribution of {selected_column}"
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

    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
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

    box_fig = px.box(
        df,
        y=box_column,
        title=f"Box Plot of {box_column}"
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

    scatter_fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        title="Feature Relationship"
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

    df = pd.read_csv(uploaded_file)

    # Simulate live data
    live_data = df.sample(200)

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

