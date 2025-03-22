# Author: Arjun Sudheer

from pathlib import Path
import joblib
import numpy as np
import optuna
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from decision_tree import DecisionTreeNetworkAttackClassifier


class AdaBoostNetworkAttackClassifier:
    def __init__(self, X_train, y_train, dataset):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset

        self.weak_learner = DecisionTreeNetworkAttackClassifier(
            X_train, y_train, dataset
        ).best_clf

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(
            f"{self.dataset}/saved_classifier_models/ada_boost_trained.pkl"
        ).exists():
            self.best_clf = joblib.load(
                f"{self.dataset}/saved_classifier_models/ada_boost_trained.pkl"
            )
        else:
            Path(f"{self.dataset}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.__train()

    def __train(self):
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
            learning_rate = trial.suggest_float("learning_rate", 0.5, 2.0, step=0.5)

            clf = AdaBoostClassifier(
                random_state=42,
                estimator=self.weak_learner,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
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
        study.optimize(objective, n_trials=10)

        # Get the best parameters
        best_params = study.best_params

        # Train the final AdaBoost model
        self.best_clf = AdaBoostClassifier(
            random_state=42,
            estimator=self.weak_learner,
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
        )
        self.best_clf.fit(self.X_train, self.y_train)

        joblib.dump(
            self.best_clf,
            f"{self.dataset}/saved_classifier_models/ada_boost_trained.pkl",
        )

    def predict_network_attack_class(self, X_test):
        return self.best_clf.predict(X_test)
