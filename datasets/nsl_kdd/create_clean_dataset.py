# Author: Arjun Sudheer

import cupy as cp
import cudf as cd
from cuml.preprocessing import StandardScaler, LabelEncoder, train_test_split
import joblib


LABEL_COLUMN = "labels"
CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag",
    "labels",
]


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


def save_dataset(df_cleaned):
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
    train_df = cd.read_csv("original_dataset/kdd_train.csv")
    test_df = cd.read_csv("original_dataset/kdd_test.csv")
    merged_df = cd.concat([train_df, test_df])

    if LABEL_COLUMN not in merged_df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in DataFrame")

    numerical_columns = [
        col for col in merged_df.columns if col not in CATEGORICAL_COLUMNS
    ]

    merged_df_cleaned = drop_duplicates_and_nan(merged_df)
    merged_df_cleaned = drop_infrequent_classes(merged_df_cleaned)
    merged_df_cleaned = drop_constant_columns(merged_df_cleaned)
    merged_df_cleaned = clean_numeric_columns(merged_df_cleaned, numerical_columns)
    save_dataset(merged_df_cleaned)
