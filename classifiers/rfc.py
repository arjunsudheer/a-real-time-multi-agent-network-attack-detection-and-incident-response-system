# Author: Arjun Sudheer

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class RFCNetworkAttackClassifier:
    def __init__(self, X_train, y_train, dataset):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(f"{self.dataset}/saved_classifier_models/rfc_trained.pkl").exists():
            self.best_clf = joblib.load(
                f"{self.dataset}/saved_classifier_models/rfc_trained.pkl"
            )
        else:
            Path(f"{self.dataset}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.__train()

    def __train(self):
        def objective(trial):
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

            # Compute cross-validated F1-score (weighted)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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
            f"{self.dataset}/saved_classifier_models/rfc_trained.pkl",
        )

    def predict_network_attack_class(self, X_test):
        return self.best_clf.predict(X_test)
