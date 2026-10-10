import pandas as pd


def load_data(file_path):
    """
    Load network data from a CSV file.
    """
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Basic preprocessing of network traffic data.
    """

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Replace infinite values
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Remove rows containing missing values
    df = df.dropna()

    return df


def preprocess(file_path):
    """
    Load and clean a dataset.
    """
    df = load_data(file_path)
    df = clean_data(df)

    return df