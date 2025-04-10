# Author: Arjun Sudheer

from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib

from classifiers.rfc import RFCNetworkAttackClassifier
from classifiers.mlp import MLPNetworkAttackClassifier
from classifiers.decision_tree import DecisionTreeNetworkAttackClassifier
from classifiers.knn import KNNNetworkAttackClassifier
from classifiers.xg_boost import XGBoostNetworkAttackClassifier


def load_data(dataset_directory: str) -> tuple[np.ndarray]:
    train_df = pd.read_csv(f"{dataset_directory}/train.csv")
    test_df = pd.read_csv(f"{dataset_directory}/test.csv")

    with open(f"{dataset_directory}/num_y_columns.txt", "r") as f:
        num_y_train_cols = int(f.readline())
        num_y_test_cols = int(f.readline())

    # Separate features and labels
    y_train = train_df.iloc[:, -num_y_train_cols:]
    X_train = train_df.iloc[:, :-num_y_train_cols]

    y_test = test_df.iloc[:, -num_y_test_cols:]
    X_test = test_df.iloc[:, :-num_y_test_cols]

    # Convert one-hot to single-label
    y_train = np.argmax(y_train.values, axis=1)
    y_test = np.argmax(y_test.values, axis=1)

    return X_train, y_train, X_test, y_test


def load_preprocessors(dataset_directory: str) -> tuple[LabelBinarizer, StandardScaler]:
    # Load the LabelBinarizer from the pickle file
    label_binarizer_path = f"{dataset_directory}/label_binarizer.pkl"
    with open(label_binarizer_path, "rb") as file:
        label_binarizer = joblib.load(file)

    # Load the StandardScaler from the pickle file
    standard_scaler_path = f"{dataset_directory}/standard_scaler.pkl"
    with open(standard_scaler_path, "rb") as file:
        standard_scaler = joblib.load(file)

    return label_binarizer, standard_scaler


def get_signature_method_classification(
    dataset_directory: str, samples: np.ndarray
) -> np.ndarray:

    X_train, y_train, _, _ = load_data(dataset_directory)
    dt = DecisionTreeNetworkAttackClassifier(X_train, y_train, dataset_directory)

    return dt.predict_network_attack_class(samples)


def get_robust_classifier_predictions(
    dataset_directory: str, samples: np.ndarray
) -> dict:
    X_train, y_train, _, _ = load_data(dataset_directory)

    rfc = RFCNetworkAttackClassifier(X_train, y_train, dataset_directory)
    xgb = XGBoostNetworkAttackClassifier(X_train, y_train, dataset_directory)
    mlp = MLPNetworkAttackClassifier(X_train, y_train, dataset_directory)
    knn = KNNNetworkAttackClassifier(X_train, y_train, dataset_directory)

    predictions = {}

    predictions["rfc"] = rfc.predict_network_attack_class(samples)
    predictions["xgb"] = xgb.predict_network_attack_class(samples)
    predictions["mlp"] = mlp.predict_network_attack_class(samples)
    predictions["knn"] = knn.predict_network_attack_class(samples)

    return predictions


def calculate_majority_classification(
    signature_method_predictions: np.ndarray, robust_classifier_predictions: dict
) -> tuple[list[int], list[bool]]:
    majority_predictions = []
    signature_and_robust_agreement = []

    # Majority voting
    for i in range(len(signature_method_predictions)):
        signature_method_prediction = signature_method_predictions[i]

        current_predictions = []
        for classifier in robust_classifier_predictions.keys():
            current_predictions.append(robust_classifier_predictions[classifier][i])

        occurrences = Counter(current_predictions)
        robust_classifier_majority_prediction = occurrences.most_common(1)[0][0]
        majority_predictions.append((robust_classifier_majority_prediction))

        signature_and_robust_agreement.append(
            signature_method_prediction == robust_classifier_majority_prediction
        )

    return majority_predictions, signature_and_robust_agreement


if __name__ == "__main__":
    dataset_directory = "datasets/aci_iot_network_dataset_2023"

    _, _, X_test, y_test = load_data(dataset_directory=dataset_directory)

    # Get classifier predictions
    signature_method_predictions = get_signature_method_classification(
        dataset_directory=dataset_directory, samples=X_test
    )
    robust_classifier_predictions = get_robust_classifier_predictions(
        dataset_directory=dataset_directory, samples=X_test
    )
    majority_predictions, _ = calculate_majority_classification(
        signature_method_predictions=signature_method_predictions,
        robust_classifier_predictions=robust_classifier_predictions,
    )

    all_predictions = {
        "dt": signature_method_predictions,
        "rfc": robust_classifier_predictions["rfc"],
        "xgb": robust_classifier_predictions["xgb"],
        "mlp": robust_classifier_predictions["mlp"],
        "knn": robust_classifier_predictions["knn"],
        "majority_predictions": majority_predictions,
    }

    # Create the directory for saving metrics, if it doesn't exist
    metrics_save_directory = Path(f"{dataset_directory}/classifier_metrics")
    metrics_save_directory.mkdir(parents=True, exist_ok=True)

    # Save classification metrics to a file
    for key in all_predictions:
        y_pred = all_predictions[key]

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(
            y_test, y_pred, zero_division=0, average="weighted"
        )
        test_recall = recall_score(y_test, y_pred, zero_division=0, average="weighted")
        test_f1_score = f1_score(y_test, y_pred, zero_division=0, average="weighted")
        test_confusion_matrix = confusion_matrix(y_test, y_pred)

        # Save accuracy, precision, recall, and confusion matrix for both train and test sets
        with open(metrics_save_directory / f"{key}.txt", "w") as f:
            f.write("Test metrics:\n")
            f.write(f"Accuracy: {test_accuracy}\n")
            f.write(f"Precision: {test_precision}\n")
            f.write(f"Recall: {test_recall}\n")
            f.write(f"F1 Score: {test_f1_score}\n")
            f.write(f"Confusion Matrix:\n{test_confusion_matrix}\n")
