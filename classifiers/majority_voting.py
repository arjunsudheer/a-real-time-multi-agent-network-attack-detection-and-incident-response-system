# Author: Arjun Sudheer

from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os

from rfc import RFCNetworkAttackClassifier
from svm import SVMNetworkAttackClassifier
from mlp import MLPNetworkAttackClassifier
from decision_tree import DecisionTreeNetworkAttackClassifier
from knn import KNNNetworkAttackClassifier
from xg_boost import XGBoostNetworkAttackClassifier
from ada_boost import AdaBoostNetworkAttackClassifier

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)


class MajorityVoting:
    def __init__(self, dataset_directory, label_column):
        self.dataset_directory = dataset_directory
        self.label_column = label_column

        self.majority_prediction_key = "majority_predictions"

        self.__load_data()

        # Initialize and train classifiers if needed
        self.rfc = RFCNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.xgb = XGBoostNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.svm = SVMNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.mlp = MLPNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.dt = DecisionTreeNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.knn = KNNNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )
        self.ab = AdaBoostNetworkAttackClassifier(
            self.X_train, self.y_train, dataset_directory
        )

    def __load_data(self):
        train_df = pd.read_csv(f"{self.dataset_directory}/train.csv")
        test_df = pd.read_csv(f"{self.dataset_directory}/test.csv")

        # Convert labels to int64 (ensure proper encoding if categorical)
        self.y_train = train_df[self.label_column].astype(np.int64).values
        self.y_test = test_df[self.label_column].astype(np.int64).values

        # Convert features to float32 (compatible with both scikit-learn and PyTorch)
        self.X_train = (
            train_df.drop(columns=[self.label_column]).astype(np.float32).values
        )
        self.X_test = (
            test_df.drop(columns=[self.label_column]).astype(np.float32).values
        )

    def __get_classifier_predictions(self, samples):
        predictions = {}

        predictions["rfc"] = self.rfc.predict_network_attack_class(samples)
        predictions["xgb"] = self.xgb.predict_network_attack_class(samples)
        predictions["svm"] = self.svm.predict_network_attack_class(samples)
        predictions["mlp"] = self.mlp.predict_network_attack_class(samples)
        predictions["dt"] = self.dt.predict_network_attack_class(samples)
        predictions["knn"] = self.knn.predict_network_attack_class(samples)
        predictions["ab"] = self.ab.predict_network_attack_class(samples)

        return predictions

    def calculate_majority_classification(self, samples):
        predictions = self.__get_classifier_predictions(samples)
        # Keep track of the majority predictions
        majority_predictions = []

        # Majority voting
        for i in range(len(samples)):
            current_predictions = []
            for key in predictions:
                current_predictions.append(predictions[key][i])

            occurrences = Counter(current_predictions)
            majority_prediction = occurrences.most_common(1)[0][0]
            # Calculates how many classifiers agreed on the majority prediction
            agreement_ratio = occurrences[majority_prediction] / len(
                current_predictions
            )

            # Prompt IDS-Agent if the agreement ratio is below a predefined threshold
            if agreement_ratio < (5 / 7):
                # Update the majority prediction with the prediction from IDS-Agent
                pass

            majority_predictions.append(majority_prediction)

        # Update predictions dictionary with majority prediction
        predictions[self.majority_prediction_key] = majority_predictions

        return predictions

    def calculate_classification_metrics(self):
        # Create the directory for saving metrics, if it doesn't exist
        metrics_save_directory = Path(f"{self.dataset_directory}/classifier_metrics")
        metrics_save_directory.mkdir(parents=True, exist_ok=True)

        # Get classifier predictions
        predictions = self.calculate_majority_classification(self.X_test)

        # Save classification metrics to a file
        for key in predictions:
            y_pred = predictions[key]

            test_accuracy = accuracy_score(self.y_test, y_pred)
            test_precision = precision_score(
                self.y_test, y_pred, zero_division=0, average="weighted"
            )
            test_recall = recall_score(
                self.y_test, y_pred, zero_division=0, average="weighted"
            )
            test_f1_score = f1_score(
                self.y_test, y_pred, zero_division=0, average="weighted"
            )
            test_confusion_matrix = confusion_matrix(self.y_test, y_pred)

            # Save accuracy, precision, recall, and confusion matrix for both train and test sets
            with open(metrics_save_directory / f"{key}.txt", "w") as f:
                f.write("Test metrics:\n")
                f.write(f"Accuracy: {test_accuracy}\n")
                f.write(f"Precision: {test_precision}\n")
                f.write(f"Recall: {test_recall}\n")
                f.write(f"F1 Score: {test_f1_score}\n")
                f.write(f"Confusion Matrix:\n{test_confusion_matrix}\n")


if __name__ == "__main__":
    datasets = {
        "datasets/nsl_kdd": "labels",
        "datasets/aci_iot_network_dataset_2023": "Label",
        "datasets/cic_iot_dataset_2023": "label",
    }

    for dataset_directory, label_column in datasets.items():
        majority_vote_classifier = MajorityVoting(dataset_directory, label_column)
        majority_vote_classifier.calculate_classification_metrics()
