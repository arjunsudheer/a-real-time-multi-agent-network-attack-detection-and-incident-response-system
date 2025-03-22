# Author: Arjun Sudheer

import cupy as cp
import cudf as cd
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import joblib


LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["label"]


def drop_duplicates_and_nan(df):
    df = df.replace([cp.inf, -cp.inf], cp.nan)
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
    cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
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


def encode_categorical_columns(df):
    label_encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            label_encoders[col] = le
            # Convert to string before encoding
            df[col] = le.fit_transform(df[col].astype(str))

    # Save the label encoders to a file
    joblib.dump(label_encoders[LABEL_COLUMN], "label_encoder.pkl")

    return df


def save_train_and_test_datasets(df_cleaned):
    # Encode categorical features
    df_encoded = encode_categorical_columns(df_cleaned)

    # Create initial train and test datasets
    X = df_encoded.drop([LABEL_COLUMN], axis=1)
    y = df_encoded[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    train_df = scaler.fit_transform(X_train)
    test_df = scaler.transform(X_test)

    # Combine scaled features with labels
    train_df[LABEL_COLUMN] = y_train.to_cupy()
    test_df[LABEL_COLUMN] = y_test.to_cupy()

    # Save train and test dataframes to separate CSV files
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print("\nClass distribution:")
    print(train_df[LABEL_COLUMN].value_counts(normalize=True))


if __name__ == "__main__":
    csv_file = list(Path(".").glob("original_dataset/*.csv"))[0]
    df = cd.read_csv(csv_file)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in DataFrame")

    numerical_columns = [col for col in df.columns if col not in CATEGORICAL_COLUMNS]

    df_cleaned = drop_duplicates_and_nan(df)
    df_cleaned = drop_infrequent_classes(df_cleaned)
    df_cleaned = drop_constant_columns(df_cleaned)
    df_cleaned = clean_numeric_columns(df_cleaned, numerical_columns)

    save_train_and_test_datasets(df_cleaned)
