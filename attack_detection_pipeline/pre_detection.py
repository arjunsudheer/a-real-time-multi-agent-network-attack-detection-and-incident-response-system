from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class PreDetection:
    def __init__(
        self, X_train: pd.DataFrame, y_train: np.ndarray, dataset_directory: Path
    ) -> None:
        self.X_train = X_train
        # Convert to binary labels
        self.y_train = np.where(y_train == "Benign", 0, 1)
        self.dataset_directory = dataset_directory

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(
            f"{self.dataset_directory}/saved_classifier_models/pre_detection_trained.pkl"
        ).exists():
            self.best_clf = joblib.load(
                f"{self.dataset_directory}/saved_classifier_models/pre_detection_trained.pkl"
            )
        else:
            Path(f"{self.dataset_directory}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.__train()

    def __train(self) -> None:
        def objective(trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 50, 150, step=50)
            max_depth = trial.suggest_categorical("max_depth", [20, 30, None])
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
            min_weight_fraction_leaf = trial.suggest_float(
                "min_weight_fraction_leaf", 0.0, 0.05, step=0.01
            )
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])
            class_weight = trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            )

            clf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,  # Run jobs in parallel
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight=class_weight,
            )

            # Compute cross-validated accuracy (balanced)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                clf,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="balanced_accuracy",
            )

            return np.mean(scores)

        # Run Optuna optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, gc_after_trial=True)

        # Train with the best parameters
        best_params = study.best_params
        self.best_clf = RandomForestClassifier(
            **best_params, random_state=42, n_jobs=-1
        )
        self.best_clf.fit(self.X_train, self.y_train)

        # Save the best model
        joblib.dump(
            self.best_clf,
            f"{self.dataset_directory}/saved_classifier_models/pre_detection_trained.pkl",
        )

    def __get_classifier_predictions(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.best_clf.predict(X_test)

    def save_classification_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        unique_labels: np.array = np.array(["Benign", "Malicious"]),
    ):
        y_pred = self.__get_classifier_predictions(X_test)
        # Convert to binary labels
        y_test = np.where(y_test == "Benign", 0, 1)

        # Create the directory for saving metrics, if it doesn't exist
        metrics_save_directory = Path(f"{self.dataset_directory}/classifier_metrics")
        metrics_save_directory.mkdir(parents=True, exist_ok=True)

        # Save classification metrics to a file
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(
            y_test, y_pred, zero_division=0, average="binary"
        )
        test_recall = recall_score(y_test, y_pred, zero_division=0, average="binary")
        test_f1_score = f1_score(y_test, y_pred, zero_division=0, average="binary")
        test_confusion_matrix = confusion_matrix(y_test, y_pred)

        # Save accuracy, precision, recall, and confusion matrix for both train and test sets
        with open(metrics_save_directory / f"pre_detection.txt", "w") as f:
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
        plt.savefig(
            metrics_save_directory / f"pre_detection_confusion_matrix_display.png"
        )
        plt.close()

    def filter_malicious_samples(
        self,
        preprocessed_samples: pd.DataFrame,
        original_samples: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # Pre-detection predictions
        predictions = self.__get_classifier_predictions(preprocessed_samples)

        # Find the malicious prediction indices
        malicious_indices = []
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                malicious_indices.append(i)

        malicious_preprocessed_samples_df = preprocessed_samples.iloc[malicious_indices]
        malicious_original_samples_df = original_samples.iloc[malicious_indices]

        return (
            malicious_preprocessed_samples_df,
            malicious_original_samples_df,
            malicious_indices,
        )
