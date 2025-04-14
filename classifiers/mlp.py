# Author: Arjun Sudheer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


class NetworkDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPNetworkAttackClassifier:
    def __init__(self, X_train, y_train, dataset):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.__build_model().to(self.device)

        # Load an already trained classifier model if it exists
        # Otherwise, create a new classifier model to train
        if Path(f"{self.dataset}/saved_classifier_models/mlp_trained.pt").exists():
            self.clf = self.model.load_state_dict(
                torch.load(
                    Path(f"{self.dataset}/saved_classifier_models") / "mlp_trained.pt"
                )
            )
        else:
            Path(f"{self.dataset}/saved_classifier_models").mkdir(
                exist_ok=True, parents=True
            )
            self.kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            self.patience = 7
            self.__train()

    def __build_model(self):
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

    def __train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

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
                self.model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
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
                    torch.save(
                        self.model.state_dict(),
                        Path(f"{self.dataset}/saved_classifier_models")
                        / "mlp_trained.pt",
                    )
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def predict_network_attack_class(self, X_test):
        self.model.load_state_dict(
            torch.load(
                Path(f"{self.dataset}/saved_classifier_models") / "mlp_trained.pt"
            )
        )
        self.model.eval()
        X_test = torch.tensor(X_test.values, dtype=torch.float32).to(self.device)
        test_loader = DataLoader(X_test, batch_size=32)
        y_pred = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        return y_pred
