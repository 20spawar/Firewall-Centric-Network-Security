import pandas as pd


def encode_categorical_features(df):
    """
    Convert categorical network features into numerical features.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False
    )

    return df


def extract_features(df, exclude_columns=None):
    """
    Prepare numerical features for anomaly detection.
    """

    df = df.copy()

    if exclude_columns is None:
        exclude_columns = []

    # Preserve only columns intended for modelling
    feature_df = df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    # Convert categorical columns
    feature_df = encode_categorical_features(feature_df)

    # Ensure everything is numeric
    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove rows that became invalid
    feature_df = feature_df.dropna()

    return feature_df