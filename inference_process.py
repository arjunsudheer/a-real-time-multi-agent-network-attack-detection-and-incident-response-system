from pathlib import Path
import pandas as pd

from preprocessing.data_cleaning import (
    clean_numeric_columns,
    transform_and_scale_features,
)


def preprocess_single_sample(X: pd.DataFrame, parent_directory: Path) -> pd.DataFrame:
    clean_numeric_columns()

    # Assume no label is provided
    return transform_and_scale_features(X, parent_directory)
