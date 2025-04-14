# Author: Arjun Sudheer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


LABEL_COLUMN = "Label"
CATEGORICAL_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Label",
    "Connection Type",
]


def drop_duplicates_and_nan(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def drop_infrequent_classes(df, min_frequency_threshold=10):
    # Get counts for each class
    class_frequencies = df[LABEL_COLUMN].value_counts()
    infrequent_classes = class_frequencies[
        class_frequencies < min_frequency_threshold
    ].index
    df = df[~df[LABEL_COLUMN].isin(infrequent_classes)]
    return df


def drop_constant_columns(df):
    # Don't check Label column for constant values
    features = [col for col in df.columns if col != LABEL_COLUMN]
    cols_to_drop = [col for col in features if df[col].nunique() == 1]
    return df.drop(columns=cols_to_drop)


# Some entries have a leading single quote that makes pandas treat it as a string instead a float
# Remove the leading single quotation for the numerical columns
def clean_numeric_columns(df, numeric_columns):
    for col in numeric_columns:
        if col in df.columns:
            # Remove quotes/commas, convert to float
            df[col] = (
                df[col].astype(str).str.lstrip("',").str.replace(",", "").astype(float)
            )
    return df


def encode_categorical_columns(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    label_encoders = {}
    df_encoded = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and col not in exclude_columns:
            le = LabelEncoder()
            label_encoders[col] = le
            # Convert to string before encoding
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded, label_encoders


def save_train_and_test_datasets(df_cleaned):
    print("\nOriginal Label distribution:")
    print(df_cleaned[LABEL_COLUMN].value_counts(normalize=True))

    # First encode the Label column separately
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_cleaned[LABEL_COLUMN].astype(str))

    # Save the label encoder
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("\nEncoded Label mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {i}")

    # Now encode other categorical features (excluding Label)
    df_encoded, _ = encode_categorical_columns(
        df_cleaned, exclude_columns=[LABEL_COLUMN]
    )

    # Split the data before scaling
    X = df_encoded.drop([LABEL_COLUMN], axis=1)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Scale the features (not the labels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrame to preserve column names
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Add labels back as integers
    train_df[LABEL_COLUMN] = y_train
    test_df[LABEL_COLUMN] = y_test

    # Save datasets
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # Save scaler for future use
    joblib.dump(scaler, "scaler.pkl")

    print("\nFinal class distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True))


if __name__ == "__main__":
    # Read only 10% of the data
    print("Loading data...")
    df = pd.read_csv(
        "raw_datasets/ACI-IoT-2023.csv", nrows=int(5970000 * 0.1)
    )  # 10% of ~5.97M rows

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in DataFrame")

    print("\nInitial shape:", df.shape)

    numerical_columns = [col for col in df.columns if col not in CATEGORICAL_COLUMNS]

    print("\nCleaning data...")
    df_cleaned = drop_duplicates_and_nan(df)
    print("After dropping duplicates and NaN:", df_cleaned.shape)

    df_cleaned = drop_infrequent_classes(df_cleaned)
    print("After dropping infrequent classes:", df_cleaned.shape)

    df_cleaned = drop_constant_columns(df_cleaned)
    print("After dropping constant columns:", df_cleaned.shape)

    df_cleaned = clean_numeric_columns(df_cleaned, numerical_columns)
    print("After cleaning numeric columns:", df_cleaned.shape)

    save_train_and_test_datasets(df_cleaned)
