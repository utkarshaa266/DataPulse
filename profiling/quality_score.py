def calculate_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]

    # Missing values
    missing_values = df.isnull().sum().sum()

    missing_percentage = (
        missing_values / total_cells
    ) * 100

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()

    duplicate_percentage = (
        duplicate_rows / len(df)
    ) * 100

    # Quality score
    quality_score = (
        100
        - missing_percentage
        - duplicate_percentage
    )

    # Prevent negative values
    quality_score = max(0, quality_score)

    return round(quality_score, 2)