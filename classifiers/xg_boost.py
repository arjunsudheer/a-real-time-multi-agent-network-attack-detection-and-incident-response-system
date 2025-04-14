# Author: Arjun Sudheer

import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold


class XGBoostNetworkAttackClassifier:
    def __init__(self, X_train, y_train, dataset):
        # Split data into training and validation sets
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = len(np.unique(self.y_train))
        self.dataset = dataset

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(f"{self.dataset}/saved_classifier_models/xgboost_trained.bin").exists():
            self.clf = xgb.Booster()
            self.clf.load_model(
                f"{self.dataset}/saved_classifier_models/xgboost_trained.bin"
            )

        else:
            Path(f"{self.dataset}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            self.__train()

    def __train(self):
        best_val_score = float("inf")
        best_fold_model = None

        for train_index, val_index in self.kf.split(self.X_train, self.y_train):
            X_train, X_val = (
                self.X_train.iloc[train_index].values,
                self.X_train.iloc[val_index].values,
            )
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]

            dtrain = xgb.DMatrix(X_train, y_train)
            dval = xgb.DMatrix(X_val, y_val)

            params = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "device": "cuda",
                "tree_method": "hist",
                "num_class": self.num_classes,
            }

            num_rounds = 100
            evallist = [(dtrain, "train"), (dval, "eval")]

            self.clf = xgb.train(
                params,
                dtrain,
                num_rounds,
                evals=evallist,
                early_stopping_rounds=10,
                verbose_eval=10,
            )

            # Calculate validation score (use the last validation score in the logs)
            val_score = self.clf.best_score

            # Save the best model based on validation score
            if val_score < best_val_score:
                best_val_score = val_score
                best_fold_model = self.clf.copy()

        # Save the trained classifier model
        best_fold_model.save_model(
            f"{self.dataset}/saved_classifier_models/xgboost_trained.bin",
        )

    def predict_network_attack_class(self, X_test):
        dtest = xgb.DMatrix(X_test.values)
        y_pred_prob = self.clf.predict(dtest)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred
