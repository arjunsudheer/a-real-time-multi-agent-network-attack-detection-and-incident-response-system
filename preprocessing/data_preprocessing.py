# Author: Arjun Sudheer

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
import joblib


def clean_data(
    df: pd.DataFrame, label_column: str, min_frequency_threshold: int = 10
) -> None:
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


# Some entries have a leading single quote that makes pandas treat it as a string instead a float
# Remove the leading single quotation for the numerical columns
def clean_numeric_columns(df: pd.DataFrame) -> None:
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


def transform_and_scale_features(
    X: pd.DataFrame, parent_directory: Path, fit_scaler: bool = False
) -> pd.DataFrame:
    # Use FeatureHasher instead of OneHotEncoder due to memory issues
    fh = FeatureHasher(n_features=1024, input_type="string")

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns

    # FeatureHash categorical columns
    X_categorical = fh.transform(X[categorical_columns].astype(str).values)
    # Convert to dense array
    X_categorical = X_categorical.toarray()
    X_numerical = X.drop(columns=categorical_columns, errors="ignore")

    # Scale numerical features
    if fit_scaler:
        scaler = StandardScaler()
        X_numerical = scaler.fit_transform(X_numerical)
        joblib.dump(scaler, parent_directory / "standard_scaler.pkl")
    else:
        scaler = joblib.load(parent_directory / "standard_scaler.pkl")
        X_numerical = scaler.transform(X_numerical)

    # Reconstruct DataFrame
    X_processed = np.hstack([X_numerical, X_categorical])
    X_df = pd.DataFrame(X_processed, index=X.index)

    return X_df


def preprocess_single_sample(X: pd.DataFrame, parent_directory: Path) -> pd.DataFrame:
    clean_numeric_columns()

    # Assume no label is provided
    return transform_and_scale_features(X, parent_directory)
