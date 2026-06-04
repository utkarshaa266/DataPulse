import plotly.express as px

def create_histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )

    return fig


def create_boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Boxplot of {column}"
    )

    return fig
def create_scatter(df, x_col, y_col):
    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        title="Feature Relationship"
    )


def create_heatmap(correlation_matrix):
    return px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )