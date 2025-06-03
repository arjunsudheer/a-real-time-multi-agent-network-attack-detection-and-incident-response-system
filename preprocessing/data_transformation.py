from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelBinarizer
import joblib


def match_column_format(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    match_column_format matches the columns and column order of a DataFrame to a reference DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        reference_df (pd.DataFrame): The reference DataFrame to use when matching the columns and column order.

    Returns:
        pd.DataFrame: The modified DataFrame with the matched columns and column order to the reference DataFrame.
    """
    # Add missing columns
    for col in reference_df.columns:
        if col not in df.columns:
            df[col] = 0.0

    # Drop extra columns
    df = df[[col for col in reference_df.columns if col in df.columns]]

    # Reorder to match reference
    df = df[reference_df.columns]

    return df


def transform_and_scale_features(
    X: pd.DataFrame, parent_directory: Path, fit_scaler: bool = False
) -> pd.DataFrame:
    """
    transform_and_scale_features transforms categorical columns to numerical columns and scales numerical columns.

    Uses the FeatureHasher to convert categorical columns to numerical values. Uses the StandardScaler to scale
    numerical values.

    Args:
        X (pd.DataFrame): The features that need to be transformed.
        parent_directory (Path): The path where the StandardScaler is saved.
        fit_scaler (bool, optional): Whether or not the StandardScaler should be fit, or an already fit StandardScaler
        should be used. Defaults to False.

    Returns:
        pd.DataFrame: The transformed features.
    """
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
    """
    load_label_binarizer loads a saved LabelBinarizer.

    Args:
        parent_directory (str): The directory where the LabelBinarizer pickle file is stored.

    Returns:
        LabelBinarizer: The saved LabelBinarizer.
    """
    # Load the LabelBinarizer from the pickle file
    label_binarizer_path = f"{parent_directory}/label_binarizer.pkl"
    with open(label_binarizer_path, "rb") as file:
        label_binarizer = joblib.load(file)

    return label_binarizer


def transform_data(
    df: pd.DataFrame,
    label_column: str,
    parent_directory: Path,
) -> list[pd.DataFrame | np.ndarray]:
    """
    transform_data splits the DataFrame into train and test DataFrame and then transforms the DataFrame.

    Uses the FeatureHasher to transform the categorical features into numerical values. Uses the
    StandardScaler to scale the numerical values. Uses the LabelBinarizer to transform the labels in a way
    that does not impose an arbitrary ordering.

    Args:
        df (pd.DataFrame): The DataFrame to transform and scale.
        label_column (str): The name of the label column to handle separately from the features.
        parent_directory (Path): The path to store the fit LabelBinarizer and StandardScaler.

    Returns:
        list[pd.DataFrame | np.ndarray]: Returns X_train, y_train, X_test, and y_test for the preprocessed DataFrames
        first, and then the original DataFrames that have just been split.
    """

    def transform_labels(
        y: pd.Series, X: pd.DataFrame, lb: LabelBinarizer
    ) -> np.ndarray:
        """
        transform_labels transforms the label column into a list of columns of 0's and 1's, similar to OneHotEncoding.

        Transforms each label into a list of label columns where 0 indicates a false value, and 1 indicates a true value.
        A value of 1 indicates the actual label of the current sample. Avoids imposing any numerical ordering on the labels.

        Args:
            y (pd.Series): The labels to transform.
            X (pd.DataFrame): The features in the dataframe. Used to align the index after transforming the labels.
            lb (LabelBinarizer): The fit LabelBinarizer that should be used for the transformation.

        Returns:
            np.ndarray: The transformed labels.
        """
        # Label binarize the label column
        y_preprocessed = lb.transform(y)

        # Match the index with the features and rename the columns created by LabelBinarizer
        y_preprocessed = pd.DataFrame(
            y_preprocessed,
            index=X.index,
            columns=[f"{label_column}_{i}" for i in range(y_preprocessed.shape[1])],
        )

        y_preprocessed = np.argmax(y_preprocessed.values, axis=1)

        return y_preprocessed

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Separate features and labels for original dataset
    X_train_original = train_df.drop(columns=[label_column]).reset_index(drop=True)
    y_train_original = train_df[label_column].reset_index(drop=True)
    X_test_original = test_df.drop(columns=[label_column]).reset_index(drop=True)
    y_test_original = test_df[label_column].reset_index(drop=True)

    # Preprocess for pre-detection
    # Transform and scale features for pre-detection
    X_train_pre_detection = transform_and_scale_features(
        X_train_original, parent_directory, fit_scaler=True
    )
    X_test_pre_detection = transform_and_scale_features(
        X_test_original, parent_directory, fit_scaler=False
    )
    # Convert labels to binary (0 = "Benign", 1 = all other malicious classes)
    y_train_pre_detection = np.where(y_train_original == "Benign", 0, 1)
    y_test_pre_detection = np.where(y_test_original == "Benign", 0, 1)

    # Preprocess for post-classification
    # Fit the label binarizer on the whole dataset excluding the "Benign" label
    lb = LabelBinarizer()
    y = df[df[label_column] != "Benign"][label_column]
    lb.fit(y)
    joblib.dump(lb, parent_directory / "label_binarizer.pkl")  # Save encoder
    # Remove the benign samples for post-classification
    train_mask = y_train_original != "Benign"
    X_train_post_classification = X_train_original[train_mask].reset_index(drop=True)
    y_train_post_classification = y_train_original[train_mask].reset_index(drop=True)
    test_mask = y_test_original != "Benign"
    X_test_post_classification = X_test_original[test_mask].reset_index(drop=True)
    y_test_post_classification = y_test_original[test_mask].reset_index(drop=True)
    # Transform and scale features for post-classification
    X_train_post_classification = transform_and_scale_features(
        X_train_post_classification, parent_directory, fit_scaler=True
    )
    X_test_post_classification = transform_and_scale_features(
        X_test_post_classification, parent_directory, fit_scaler=False
    )
    # Transform labels for post-classification
    y_train_post_classification = transform_labels(
        y=y_train_post_classification, X=X_train_post_classification, lb=lb
    )
    y_test_post_classification = transform_labels(
        y=y_test_post_classification,
        X=X_test_post_classification,
        lb=lb,
    )

    return [
        X_train_original,
        y_train_original.to_numpy(),
        X_test_original,
        y_test_original.to_numpy(),
        X_train_pre_detection,
        y_train_pre_detection,
        X_test_pre_detection,
        y_test_pre_detection,
        X_train_post_classification,
        y_train_post_classification,
        X_test_post_classification,
        y_test_post_classification,
    ]
