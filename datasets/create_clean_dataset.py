# Author: Arjun Sudheer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_extraction import FeatureHasher
import joblib
from pathlib import Path


class NetworkAttackDataset:
    def __init__(
        self, df, label_column, categorical_columns, parent_directory, sampling_rate=0.1
    ):
        self.df = df.sample(frac=sampling_rate, random_state=42)
        self.label_column = label_column
        self.categorical_columns = categorical_columns
        self.parent_directory = parent_directory

        if self.label_column not in self.df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in DataFrame"
            )

        # Use sklearn and numpy since cupy does not support string arrays
        self.lb = LabelBinarizer()
        # Use FeatureHasher instead of OneHotEncoder due to memory issues
        self.fh = FeatureHasher(n_features=1024, input_type="string")
        # Use sklearn StandardScaler instead of cupy to avoid cupy array conversion
        self.scaler = StandardScaler()

    def __clean_data(self):
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        # Drop constant columns
        self.df.drop(
            columns=[col for col in self.df.columns if self.df[col].nunique() == 1],
            inplace=True,
        )

    def __drop_infrequent_classes(self, min_frequency_threshold=10):
        # Get counts for each class
        class_counts = self.df[self.label_column].value_counts()
        self.df = self.df[
            self.df[self.label_column].isin(
                class_counts[class_counts >= min_frequency_threshold].index
            )
        ]

    # Some entries have a leading single quote that makes pandas treat it as a string instead a float
    # Remove the leading single quotation for the numerical columns
    def __clean_numeric_columns(self):
        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        self.df[numeric_columns] = (
            self.df[numeric_columns]
            .astype(str)
            .replace({",": "", "'": ""}, regex=True)
            .astype(float)
        )

    def __transform_data(self, df, fit_scaler):
        X = df.drop(columns=[self.label_column])
        # Label binarize label column
        y = self.lb.transform(df[self.label_column])

        # FeatureHash categorical columns
        X_categorical = self.fh.transform(
            X[self.categorical_columns].astype(str).values
        )
        # Convert to dense array
        X_categorical = X_categorical.toarray()
        X_numerical = X.drop(columns=self.categorical_columns, errors="ignore")

        # Scale numerical features
        if fit_scaler:
            X_numerical = self.scaler.fit_transform(X_numerical)
            joblib.dump(self.scaler, self.parent_directory / "standard_scaler.pkl")
        else:
            X_numerical = self.scaler.transform(X_numerical)

        # Reconstruct DataFrame
        X_processed = np.hstack([X_numerical, X_categorical])
        X_df = pd.DataFrame(X_processed, index=df.index)
        y_df = pd.DataFrame(
            y,
            index=df.index,
            columns=[f"{self.label_column}_{i}" for i in range(y.shape[1])],
        )

        # Save the number of label columns to a text file
        with open(self.parent_directory / "num_y_columns.txt", "a") as f:
            f.write(f"{y_df.shape[1]}\n")

        return pd.concat([X_df, y_df], axis=1)

    def __save_dataset(
        self, df, save_file, fit_scaler=False, save_original=True, save_transformed=True
    ):
        if save_original:
            # Save original csv file for use with the llm
            df.to_csv(self.parent_directory / f"original_{save_file}.csv")

        if save_transformed:
            # Save proprocessed dataframe
            df_preprocessed = self.__transform_data(df, fit_scaler)
            df_preprocessed.to_csv(
                self.parent_directory / f"{save_file}.csv", index=False
            )

    def preprocess_datasets(self):
        self.__clean_data()
        self.__drop_infrequent_classes()
        self.__clean_numeric_columns()

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        num_samples_for_llm_ltm_cache = 100
        test_llm_ltm_split = 1 - (num_samples_for_llm_ltm_cache / len(test_df))
        llm_ltm_df, test_df = train_test_split(
            test_df, test_size=test_llm_ltm_split, random_state=42
        )

        # Fit the label binarizer on the whole dataset
        y = self.df[self.label_column]
        self.lb.fit(y)
        # Save encoder
        joblib.dump(self.lb, self.parent_directory / "label_binarizer.pkl")

        self.__save_dataset(train_df, "train", fit_scaler=True, save_original=False)
        self.__save_dataset(llm_ltm_df, "llm_ltm", save_transformed=False)
        self.__save_dataset(test_df, "test", save_original=False)


if __name__ == "__main__":
    # NSL KDD dataset
    train_df = pd.read_csv("nsl_kdd/original_dataset/kdd_train.csv")
    test_df = pd.read_csv("nsl_kdd/original_dataset/kdd_test.csv")
    merged_df = pd.concat([train_df, test_df])

    nsl_kdd_nad = NetworkAttackDataset(
        df=merged_df,
        label_column="labels",
        categorical_columns=[
            "protocol_type",
            "service",
            "flag",
        ],
        parent_directory=Path("nsl_kdd"),
    )
    nsl_kdd_nad.preprocess_datasets()

    # CIC IOT Dataset 2023
    csv_file = list(Path(".").glob("cic_iot_dataset_2023/original_dataset/*.csv"))[0]
    df = pd.read_csv(csv_file)

    cic_iot_nad = NetworkAttackDataset(
        df=df,
        label_column="label",
        categorical_columns=[],
        parent_directory=Path("cic_iot_dataset_2023"),
    )
    cic_iot_nad.preprocess_datasets()

    # ACI IOT Dataset 2023
    df = pd.read_csv("aci_iot_network_dataset_2023/original_dataset/ACI-IoT-2023.csv")

    aci_iot_nad = NetworkAttackDataset(
        df=df,
        label_column="Label",
        categorical_columns=[
            "Flow ID",
            "Src IP",
            "Dst IP",
            "Timestamp",
            "Connection Type",
        ],
        parent_directory=Path("aci_iot_network_dataset_2023"),
    )
    aci_iot_nad.preprocess_datasets()
