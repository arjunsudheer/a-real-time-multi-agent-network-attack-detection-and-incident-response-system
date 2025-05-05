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
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

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
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, dataset_directory: Path
    ):
        """
        __init__ initializes the multiclass classifiers and the dataset directory.

        Args:
            X_train (pd.DataFrame): The train dataset samples.
            y_train (pd.DataFrame): The train dataset labels.
            dataset_directory (Path): The parent directory to store the classifier weights in.
        """
        self.rfc = RFCNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.xgb = XGBoostNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.mlp = MLPNetworkAttackClassifier(X_train, y_train, dataset_directory)
        self.knn = KNNNetworkAttackClassifier(X_train, y_train, dataset_directory)

        self.dataset_directory = dataset_directory

    def __get_classifier_predictions(self, samples: pd.DataFrame) -> tuple[dict, list]:
        """
        __get_classifier_predictions gets the predictions from all multiclass classifiers and calculates the majority prediction.

        Args:
            samples (pd.DataFrame): The samples to get the classifier predictions on.

        Returns:
            tuple[dict, list]: The predictions from each classifier and the calculated majority prediction. The agreement ratio for the majority sample prediction.
        """
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

    def __get_classifier_prediction_probabilities(self, samples: pd.DataFrame) -> dict:
        """
        __get_classifier_prediction_probabilities gets the class prediction probabilities from all multiclass classifiers and calculates the mean class probability for the majority vote.

        Args:
            samples (pd.DataFrame): The sample to get the classifier prediction probability on.

        Returns:
            dict: The prediction probabilities from each classifier and the mean probability for the majority vote.
        """
        prediction_probabilities = {}

        prediction_probabilities["rfc"] = (
            self.rfc.predict_network_attack_class_probabilities(samples)
        )
        prediction_probabilities["xgb"] = (
            self.xgb.predict_network_attack_class_probabilities(samples)
        )
        prediction_probabilities["mlp"] = (
            self.mlp.predict_network_attack_class_probabilities(samples)
        )
        prediction_probabilities["knn"] = (
            self.knn.predict_network_attack_class_probabilities(samples)
        )
        # Stack the other classifier predictions into a 3D array of shape (n_classifiers, n_samples, n_classes)
        stacked_probs = np.array(
            [
                prediction_probabilities["rfc"],
                prediction_probabilities["xgb"],
                prediction_probabilities["mlp"],
                prediction_probabilities["knn"],
            ]
        )

        # Mean over classifiers of shape (n_samples, n_classes)
        prediction_probabilities["majority_predictions"] = np.mean(
            stacked_probs, axis=0
        )

        return prediction_probabilities

    def save_classification_metrics(
        self, X_test: pd.DataFrame, y_test: np.ndarray, unique_labels: np.ndarray
    ):
        """
        save_classification_metrics saves the classifier metrics to a directory.

        Saves the classifier accuracy, precision, recall, f1-score, and confusion matrix results on the test dataset to a file. Generates a Confusion Matrix Display and a Precision-Recall curve.

        Args:
            X_test (pd.DataFrame): The test samples to make predictions on.
            y_test (np.ndarray): The true test labels to use for evaluation.
            unique_labels (np.ndarray): A list of the unique labels that exist in the test dataset. Used for creating the Confusion Matrix Display.
        """
        predictions, _ = self.__get_classifier_predictions(X_test)

        # Get the probabilities needed to plot the precision-recall curves
        y_test_bin = label_binarize(y_test, classes=np.arange(len(unique_labels)))
        y_scores = self.__get_classifier_prediction_probabilities(X_test)

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

            # Save a precision-recall curve
            for i, class_label in enumerate(np.unique(y_test)):
                if y_test_bin.shape[1] > i and y_scores[key].shape[1] > i:
                    precision, recall, _ = precision_recall_curve(
                        y_test_bin[:, i], y_scores[key][:, i]
                    )
                    plt.plot(recall, precision, label=unique_labels[class_label])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{key.upper()} Precision-Recall Curve")
            plt.legend(loc="best")
            plt.savefig(metrics_save_directory / f"{key}_precision_recall_curve.png")
            plt.close()

    def filter_low_agreement_samples(
        self,
        preprocessed_malicious_samples: pd.DataFrame,
        original_malicious_samples: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        filter_low_agreement_samples identifies any samples that do not have a clear majority class prediction.

        Args:
            preprocessed_malicious_samples (pd.DataFrame): The preprocessed and transformed samples to make the classifier predictions on.
            original_malicious_samples (pd.DataFrame): The original un-transformed samples to filter for use with the labeling agent later on if needed.

        Returns:
            tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]: The classifier predictions, low agreement preprocessed samples, low agreement original samples, and the low agreement sample indices.
        """
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
