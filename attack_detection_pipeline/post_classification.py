from collections import Counter
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from attack_detection_pipeline.multiclass_classifiers.rfc import (
    RFCNetworkAttackClassifier,
)
from attack_detection_pipeline.multiclass_classifiers.mlp import (
    MLPNetworkAttackClassifier,
)
from attack_detection_pipeline.multiclass_classifiers.knn import (
    KNNNetworkAttackClassifier,
)
from attack_detection_pipeline.multiclass_classifiers.xg_boost import (
    XGBoostNetworkAttackClassifier,
)


class PostClassification:
    def __init__(self, X_train, y_train: pd.DataFrame, dataset_directory: Path):
        self.rfc = RFCNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.xgb = XGBoostNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.mlp = MLPNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.knn = KNNNetworkAttackClassifier(X_train, y_train, dataset_directory)

        self.dataset_directory = dataset_directory

    def __get_classifier_predictions(self, samples: pd.DataFrame) -> dict:
        majority_predictions = []
        agreement_ratios = []
        predictions = {}

        predictions["rfc"] = self.rfc.predict_network_attack_class(samples)
        predictions["xgb"] = self.xgb.predict_network_attack_class(samples)
        predictions["mlp"] = self.mlp.predict_network_attack_class(samples)
        predictions["knn"] = self.knn.predict_network_attack_class(samples)

        num_post_detection_classifiers = len(predictions.keys())

        # Majority predictions
        for i in range(len(predictions["rfc"])):
            current_predictions = []
            for classifier in predictions.keys():
                current_predictions.append(predictions[classifier][i])

            occurrences = Counter(current_predictions)
            majority = occurrences.most_common(1)[0]
            majority_predictions.append(majority[0])

            agreement_ratios.append(majority[1] / num_post_detection_classifiers)

        predictions["majority_predictions"] = np.array(majority_predictions)

        return predictions, agreement_ratios

    def save_classification_metrics(
        self, X_test: pd.DataFrame, y_test: np.ndarray, unique_labels: np.ndarray
    ):
        predictions, _ = self.__get_classifier_predictions(X_test)
        # Create the directory for saving metrics, if it doesn't exist
        metrics_save_directory = Path(f"{self.dataset_directory}/classifier_metrics")
        metrics_save_directory.mkdir(parents=True, exist_ok=True)

        # Save classification metrics to a file
        for key in predictions.keys():
            y_pred = predictions[key]

            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(
                y_test, y_pred, zero_division=0, average="weighted"
            )
            test_recall = recall_score(
                y_test, y_pred, zero_division=0, average="weighted"
            )
            test_f1_score = f1_score(
                y_test, y_pred, zero_division=0, average="weighted"
            )
            test_confusion_matrix = confusion_matrix(y_test, y_pred)

            # Save accuracy, precision, recall, and confusion matrix for both train and test sets
            with open(metrics_save_directory / f"{key}.txt", "w") as f:
                f.write("Test metrics:\n")
                f.write(f"Accuracy: {test_accuracy}\n")
                f.write(f"Precision: {test_precision}\n")
                f.write(f"Recall: {test_recall}\n")
                f.write(f"F1 Score: {test_f1_score}\n")
                f.write(f"Confusion Matrix:\n{test_confusion_matrix}\n")

            # Save a confusion matrix display
            _, ax = plt.subplots(figsize=(12, 10))
            disp = ConfusionMatrixDisplay(
                test_confusion_matrix, display_labels=unique_labels
            )
            disp.plot(cmap="Blues", ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(metrics_save_directory / f"{key}_confusion_matrix_display.png")
            plt.close()

    def filter_low_agreement_samples(
        self,
        preprocessed_malicious_samples: pd.DataFrame,
        original_malicious_samples: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
        # Post-classification predictions
        predictions, agreement_ratios = self.__get_classifier_predictions(
            preprocessed_malicious_samples
        )

        # Find the low agreement ratio sample indices
        low_agreement_sample_indices = []
        for i in range(len(agreement_ratios)):
            agreement_ratio = agreement_ratios[i]
            if agreement_ratio <= 0.5:
                low_agreement_sample_indices.append(i)

        low_agreement_preprocessed_samples_df = preprocessed_malicious_samples.iloc[
            low_agreement_sample_indices
        ]
        low_agreement_original_samples_df = original_malicious_samples.iloc[
            low_agreement_sample_indices
        ]

        return (
            predictions,
            low_agreement_preprocessed_samples_df,
            low_agreement_original_samples_df,
            low_agreement_sample_indices,
        )
