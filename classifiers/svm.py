# Author: Arjun Sudheer

from pathlib import Path
import joblib
import numpy as np
import cupy as cp
import optuna
from cuml.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


class SVMNetworkAttackClassifier:
    def __init__(self, X_train, y_train, dataset):
        self.X_train = cp.asarray(X_train)
        self.y_train = cp.asarray(y_train)
        self.dataset = dataset

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        model_path = f"{self.dataset}/saved_classifier_models/svm_trained.pkl"
        if Path(model_path).exists():
            self.best_clf = joblib.load(model_path)
        else:
            Path(f"{self.dataset}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.__train()

    def __train(self):
        def objective(trial):
            C = trial.suggest_float("C", 0.01, 2.0, log=True)
            gamma = trial.suggest_categorical("gamma", ["auto", "scale"])

            clf = SVC(
                random_state=42,
                class_weight="balanced",
                C=C,
                gamma=gamma,
            )

            # Compute cross-validated F1-score (weighted)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(
                clf,
                self.X_train.get(),
                self.y_train.get(),
                cv=cv,
                scoring="f1_weighted",
            )
            return np.mean(scores)

        # Run Optuna optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, gc_after_trial=True)

        # Train with the best parameters
        best_params = study.best_params
        self.best_clf = SVC(
            **best_params,
            random_state=42,
            class_weight="balanced",
        )
        self.best_clf.fit(self.X_train, self.y_train)

        # Save the trained classifier model
        joblib.dump(
            self.best_clf, f"{self.dataset}/saved_classifier_models/svm_trained.pkl"
        )

    def predict_network_attack_class(self, X_test):
        return self.best_clf.predict(cp.asarray(X_test)).get()
