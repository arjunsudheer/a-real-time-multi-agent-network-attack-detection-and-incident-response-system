from pathlib import Path
import joblib
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


class KNNNetworkAttackClassifier:
    def __init__(
        self, X_train: pd.DataFrame, y_train: np.ndarray, dataset_directory: Path
    ) -> None:
        """
        __init__ initializes train dataset and dataset directory.

        Trains a new KNN classifier if no saved weight was found. If a saved weight was found, the the pretrained classifier is loaded.

        Args:
            X_train (pd.DataFrame): The train samples.
            y_train (np.ndarray): The train labels.
            dataset_directory (Path): The parent directory to store the classifier weights.
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.dataset_directory = dataset_directory

        # Load an already trained classifier model if it exists
        model_path = f"{self.dataset_directory}/saved_classifier_models/knn_trained.pkl"
        if Path(model_path).exists():
            self.best_clf = joblib.load(model_path)
        else:
            Path(f"{self.dataset_directory}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.__train()

    def __train(self) -> None:
        """
        __train trains the KNN classifier using Optuna for hyperparameter tuning.
        """

        def objective(trial: optuna.Trial) -> float:
            """
            objective trains the KNN classifier using Optuna.

            Evaluates classifier performance using the weighted-F1 score.

            Args:
                trial (optuna.Trial): An Optuna trial instance.

            Returns:
                float: the mean cross validation weighted f1-score.
            """
            n_neighbors = trial.suggest_int("n_neighbors", 3, 9, step=3)
            metric = trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
            )

            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
            )

            # Compute cross-validated F1-score (weighted)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                clf,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring="f1_weighted",
            )

            return np.mean(scores)

        # Run Optuna optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, gc_after_trial=True)

        # Train with the best parameters
        best_params = study.best_params
        self.best_clf = KNeighborsClassifier(**best_params)
        self.best_clf.fit(self.X_train, self.y_train)

        # Save the trained classifier model
        joblib.dump(
            self.best_clf,
            f"{self.dataset_directory}/saved_classifier_models/knn_trained.pkl",
        )

    def predict_network_attack_class(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        predict_network_attack_class predicts the attack class on the provided samples.

        Args:
            X_test (pd.DataFrame): The samples to make predictions on.

        Returns:
            np.ndarray: The KNN classifier predictions.
        """
        return self.best_clf.predict(np.asarray(X_test))

    def predict_network_attack_class_probabilities(
        self, X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        predict_network_attack_class_probabilities predicts the attack class probabilities on the provided samples.

        Args:
            X_test (pd.DataFrame): The samples to make predictions on.

        Returns:
            np.ndarray: The KNN classifier predictions.
        """
        return self.best_clf.predict_proba(np.asarray(X_test))
