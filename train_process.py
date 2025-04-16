import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import joblib
from pathlib import Path

from preprocessing.data_cleaning import (
    clean_data,
    clean_numeric_columns,
    transform_and_scale_features,
)
from agents.feature_selection_agent import FeatureSelectionAgent


def preprocess_dataset(
    df: pd.DataFrame,
    label_column: str,
    parent_directory: Path,
    sample_size: float = 0.1,
) -> None:
    def transform_data() -> list[pd.DataFrame]:
        # Fit the label binarizer on the whole dataset
        lb = LabelBinarizer()
        y = df[label_column]
        lb.fit(y)
        # Save encoder
        joblib.dump(lb, parent_directory / "label_binarizer.pkl")

        # Split dataset
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        X_train = train_df.drop(columns=[label_column])
        X_test = test_df.drop(columns=[label_column])

        X_train_df = transform_and_scale_features(
            X_train, parent_directory, fit_scaler=True
        )
        X_test_df = transform_and_scale_features(
            X_test, parent_directory, fit_scaler=False
        )

        # Label binarize the label column
        y_train = lb.transform(train_df[label_column])
        y_test = lb.transform(test_df[label_column])

        y_train_df = pd.DataFrame(
            y_train,
            index=X_train_df.index,
            columns=[f"{label_column}_{i}" for i in range(y_train.shape[1])],
        )
        y_test_df = pd.DataFrame(
            y_test,
            index=X_test_df.index,
            columns=[f"{label_column}_{i}" for i in range(y_test.shape[1])],
        )

        # Save the number of label columns to a text file
        with open(parent_directory / "num_y_columns.txt", "w") as f:
            f.write(f"{y_train.shape[1]}\n")
            f.write(f"{y_test.shape[1]}\n")

        return [
            pd.concat([X_train_df, y_train_df], axis=1),
            train_df,
            pd.concat([X_test_df, y_test_df], axis=1),
            test_df,
        ]

    df = df.sample(frac=sample_size, random_state=42)

    fsa = FeatureSelectionAgent(
        df=df, label_column=label_column, dataset_name=parent_directory.name
    )
    df = fsa.select_features()

    df = clean_data(df, label_column)
    df = clean_numeric_columns(df)

    return transform_data()


if __name__ == "__main__":
    # ACI IOT Dataset 2023
    df = pd.read_csv(
        "datasets/aci_iot_network_dataset_2023/original_dataset/ACI-IoT-2023.csv"
    )
    train_preprocessed, train_original, test_preprocessed, test_original = (
        preprocess_dataset(
            df,
            label_column="Label",
            parent_directory=Path("datasets/aci_iot_network_dataset_2023"),
        )
    )
