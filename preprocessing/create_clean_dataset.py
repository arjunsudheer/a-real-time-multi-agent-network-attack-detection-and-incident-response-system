# Author: Arjun Sudheer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import joblib
from pathlib import Path

from preprocessing.data_preprocessing import (
    clean_data,
    clean_numeric_columns,
    transform_and_scale_features,
)
from llm_agents.feature_selection_agent import FeatureSelectionAgent


def save_dataset(
    df: pd.DataFrame,
    label_column: str,
    parent_directory: Path,
    save_file: str,
    fit_scaler: bool = False,
    save_transformed: bool = True,
) -> None:
    def transform_data() -> pd.DataFrame:
        lb = joblib.load(parent_directory / "label_binarizer.pkl")

        X = df.drop(columns=[label_column])
        # Label binarize label column
        y = lb.transform(df[label_column])

        X_df = transform_and_scale_features(X, parent_directory, fit_scaler)

        y_df = pd.DataFrame(
            y,
            index=X_df.index,
            columns=[f"{label_column}_{i}" for i in range(y.shape[1])],
        )

        # Save the number of label columns to a text file
        with open(parent_directory / "num_y_columns.txt", "a") as f:
            f.write(f"{y_df.shape[1]}\n")

        return pd.concat([X_df, y_df], axis=1)

    # Save proprocessed dataframe
    df_preprocessed = transform_data()
    df_preprocessed.to_csv(parent_directory / f"{save_file}.csv", index=False)


def preprocess_datasets(
    df: pd.DataFrame,
    label_column: str,
    parent_directory: Path,
    sample_size: float = 0.1,
) -> None:
    df = df.sample(frac=sample_size, random_state=42)

    fsa = FeatureSelectionAgent(
        df=df, label_column=label_column, dataset_name=parent_directory.name
    )
    df = fsa.select_features()

    df = clean_data(df, label_column)
    df = clean_numeric_columns(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Fit the label binarizer on the whole dataset
    lb = LabelBinarizer()
    y = df[label_column]
    lb.fit(y)
    # Save encoder
    joblib.dump(lb, parent_directory / "label_binarizer.pkl")

    save_dataset(
        train_df,
        label_column,
        parent_directory,
        "train",
        fit_scaler=True,
        save_original=False,
    )
    save_dataset(test_df, label_column, parent_directory, "test", save_original=False)


if __name__ == "__main__":
    # ACI IOT Dataset 2023
    df = pd.read_csv(
        "datasets/aci_iot_network_dataset_2023/original_dataset/ACI-IoT-2023.csv"
    )
    preprocess_datasets(
        df,
        label_column="Label",
        parent_directory=Path("datasets/aci_iot_network_dataset_2023"),
    )
