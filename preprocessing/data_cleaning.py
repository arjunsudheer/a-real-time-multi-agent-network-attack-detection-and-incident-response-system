import numpy as np
import pandas as pd


# Some entries have a leading single quote that makes pandas treat it as a string instead a float
# Remove the leading single quotation for the numerical columns
def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean_numeric_columns removes any non-numerical characters from a numerical column.

    Args:
        df (pd.DataFrame): The DataFrame to clean the numeric columns on.

    Returns:
        pd.DataFrame: The DataFrame with the cleaned numeric columns.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Work on a copy of the dataframe
    df = df.copy()

    # Clean and convert numeric columns
    df.loc[:, numeric_columns] = (
        df[numeric_columns]
        .astype(str)
        .replace({",": "", "'": ""}, regex=True)
        .astype(float)
    )

    return df


def clean_data(
    df: pd.DataFrame, label_column: str, min_frequency_threshold: int = 10
) -> pd.DataFrame:
    """
    clean_data removes infrequent labels, and samples with incomplete or useless data.

    Removes duplicate samples and samples that contain NaN or infinity values. Remove features
    that have a constant value for all samples. Remove infrequent classes that will raise errors
    with Stratified KFold Cross Validation.

    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.
        label_column (str): The name of the label column to check for infrequent classes.
        min_frequency_threshold (int, optional): The minimum number of occurrences for each sample to have.
        Samples with fewer occurrences than min_frequency_threshold will be dropped. Defaults to 10.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove NaN, infinity, and duplicate values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Drop constant columns
    df.drop(
        columns=[col for col in df.columns if df[col].nunique() == 1],
        inplace=True,
    )

    # Drop infrequent classes with fewer than min_frequency_threshold samples
    # Get counts for each class
    class_counts = df[label_column].value_counts()
    df = df[
        df[label_column].isin(
            class_counts[class_counts >= min_frequency_threshold].index
        )
    ]

    return df


def keep_selected_features(
    df: pd.DataFrame, features_to_keep: list[str], label_column: str
) -> pd.DataFrame:
    """
    keep_selected_features keeps only the specified features and ensures that the label
    column is retained.

    Args:
        df (pd.DataFrame): The DataFrame to drop the features from.
        features_to_keep (list[str]): The list of feature names to keep.
        label_column (str): The name of the label column. Used to ensure that the label
        column is always retained.

    Returns:
        pd.DataFrame: The dataframe with only the desired columns.
    """
    # Ensure that the label column is never removed
    columns_to_keep = set(features_to_keep)
    columns_to_keep.add(label_column)

    # Find the valid columns that exist in the DataFrame, and the columns_to_keep set
    valid_columns = list(columns_to_keep.intersection(df.columns))

    return df[valid_columns]
