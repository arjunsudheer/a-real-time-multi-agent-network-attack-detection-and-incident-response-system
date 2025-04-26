from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelBinarizer
import joblib

from agents.feature_selection_agent import FeatureSelectionAgent


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


def load_label_binarizer(parent_directory: str) -> LabelBinarizer:
    # Load the LabelBinarizer from the pickle file
    label_binarizer_path = f"{parent_directory}/label_binarizer.pkl"
    with open(label_binarizer_path, "rb") as file:
        label_binarizer = joblib.load(file)

    return label_binarizer


def preprocess_dataset(
    df: pd.DataFrame,
    label_column: str,
    parent_directory: Path,
    sample_size: float = 0.1,
) -> None:
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

    def transform_labels(
        y: pd.Series, X: pd.DataFrame, label_column: str, lb: LabelBinarizer
    ) -> np.ndarray:
        # Label binarize the label column
        y_preprocessed = lb.transform(y)

        y_preprocessed = pd.DataFrame(
            y_preprocessed,
            index=X.index,
            columns=[f"{label_column}_{i}" for i in range(y_preprocessed.shape[1])],
        )

        y_preprocessed = np.argmax(y_preprocessed.values, axis=1)

        return y_preprocessed

    def transform_data() -> list[pd.DataFrame | np.ndarray]:
        # Fit the label binarizer on the whole dataset
        lb = LabelBinarizer()
        y = df[label_column]
        lb.fit(y)
        # Save encoder
        joblib.dump(lb, parent_directory / "label_binarizer.pkl")

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        X_train = train_df.drop(columns=[label_column])
        y_train = train_df[label_column]
        X_test = test_df.drop(columns=[label_column])
        y_test = test_df[label_column]

        # Transform and scale features
        X_train_preprocessed = transform_and_scale_features(
            X_train, parent_directory, fit_scaler=True
        )
        X_test_preprocessed = transform_and_scale_features(
            X_test, parent_directory, fit_scaler=False
        )

        # Transform labels
        y_train_preprocessed = transform_labels(
            y=y_train, X=X_train_preprocessed, label_column="Label", lb=lb
        )
        y_test_preprocessed = transform_labels(
            y=y_test, X=X_test_preprocessed, label_column="Label", lb=lb
        )

        # Save train dataset to know the expected format for the classifiers
        X_train_preprocessed.to_csv(parent_directory / "classifier_data_format.csv")

        return [
            X_train_preprocessed,
            y_train_preprocessed,
            X_test_preprocessed,
            y_test_preprocessed,
            X_train,
            y_train,
            X_test,
            y_test,
        ]

    # Only use a portion of the dataset to improve speed and satisfy memory limitations
    df = df.sample(frac=sample_size, random_state=42)

    # Drop irrelevant features
    # fsa = FeatureSelectionAgent(
    #     df=df, label_column=label_column, dataset_name=parent_directory.name
    # )
    # df = fsa.select_features()

    # Clean data
    df = clean_data(df, label_column)
    # Remove non-numeric characters from numeric columns
    df = clean_numeric_columns(df)

    # Split dataset into train and test
    # Use the FeatureHasher, StandardScaler, and LabelBinarizer to transform the datasets
    return transform_data()
