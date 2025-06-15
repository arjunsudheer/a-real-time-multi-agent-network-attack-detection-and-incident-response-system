import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


class NetworkDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        __init__ initializes the features and labels in the Network Dataset.

        Args:
            features (torch.Tensor): The features in the dataset.
            labels (torch.Tensor): The labels in the dataset.
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """
        __len__ returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        __getitem__ returns the features and label at the specified index.

        Args:
            idx (int): The index of the desired features and label.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The features and label at the specified index.
        """
        return self.features[idx], self.labels[idx]


class MLPNetworkAttackClassifier:
    def __init__(
        self, X_train: pd.DataFrame, y_train: np.ndarray, dataset_directory: Path
    ):
        """
        __init__ initializes train dataset and dataset directory.

        Trains a new MLP classifier if no saved weight was found. If a saved weight was found, the the pretrained classifier is loaded.

         Args:
            X_train (pd.DataFrame): The train samples.
            y_train (np.ndarray): The train labels.
            dataset_directory (Path): The parent directory to store the classifier weights.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_directory = dataset_directory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(
            f"{self.dataset_directory}/saved_classifier_models/mlp_trained.pt"
        ).exists():
            self.best_clf = self.__build_model().to(self.device)
            self.best_clf.load_state_dict(
                torch.load(
                    Path(f"{self.dataset_directory}/saved_classifier_models")
                    / "mlp_trained.pt",
                    map_location=self.device,
                )
            )
        else:
            self.clf = self.__build_model().to(self.device)
            Path(f"{self.dataset_directory}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            self.patience = 7
            self.__train()

    def __build_model(self) -> nn.Sequential:
        """
        __build_model creates a new MLP model in PyTorch.

        Returns:
            nn.Sequential: The MLP model.
        """
        # Dynamically calculate the number of classes
        num_classes = len(np.unique(self.y_train))

        return nn.Sequential(
            nn.Linear(self.X_train.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def __train(self) -> None:
        """
        __train trains the MLP model.

        Uses CrossEntropyLoss for multi-class data. Uses a patience of 7 for early stopping.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.clf.parameters(), lr=1e-4)

        overall_best_val_loss = float("inf")
        best_model_weights = None

        for train_index, val_index in self.kf.split(self.X_train, self.y_train):
            X_train_fold = torch.tensor(
                self.X_train.iloc[train_index].values, dtype=torch.float32
            ).to(self.device)
            y_train_fold = torch.tensor(self.y_train[train_index], dtype=torch.long).to(
                self.device
            )
            X_val_fold = torch.tensor(
                self.X_train.iloc[val_index].values, dtype=torch.float32
            ).to(self.device)
            y_val_fold = torch.tensor(self.y_train[val_index], dtype=torch.long).to(
                self.device
            )

            train_loader = DataLoader(
                NetworkDataset(X_train_fold, y_train_fold),
                batch_size=512,
                shuffle=True,
            )
            val_loader = DataLoader(
                NetworkDataset(X_val_fold, y_val_fold), batch_size=32
            )

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(100):
                self.clf.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.clf(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.clf.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.clf(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                print(
                    f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save the overall best model across all folds
                    if val_loss < overall_best_val_loss:
                        overall_best_val_loss = val_loss
                        best_model_weights = self.clf.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        # Save the best MLP classifier weights to a file
        torch.save(
            best_model_weights,
            Path(f"{self.dataset_directory}/saved_classifier_models")
            / "mlp_trained.pt",
        )

        # Load the weights for inference
        self.best_clf = self.__build_model().to(self.device)
        self.best_clf.load_state_dict(best_model_weights)

    def predict_network_attack_class(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        predict_network_attack_class predicts the attack class on the provided samples.

        Args:
            X_test (pd.DataFrame): The samples to make predictions on.

        Returns:
            np.ndarray: The MLP classifier predictions.
        """
        self.best_clf.eval()
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        test_loader = DataLoader(X_test, batch_size=32)
        y_pred = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.best_clf(inputs)
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        return np.array(y_pred)

    def predict_network_attack_class_probabilities(
        self, X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        predict_network_attack_class_probabilities predicts the attack class probabilities on the provided samples.

        Args:
            X_test (pd.DataFrame): The samples to make predictions on.

        Returns:
            np.ndarray: The MLP classifier predictions.
        """
        self.best_clf.eval()
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        test_loader = DataLoader(X_test, batch_size=32)
        y_pred_proba = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.best_clf(inputs)
                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                y_pred_proba.extend(probabilities)

        y_pred_proba = np.array(y_pred_proba)
        return y_pred_proba
