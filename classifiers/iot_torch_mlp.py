import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time


class IoTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def plot_confusion_matrix(
    conf_matrix, class_names, output_file="visualizations/mlp/confusion_matrix.png"
):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("MLP Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_training_history(
    train_losses, val_losses, output_file="visualizations/mlp/training_history.png"
):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_mlp():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    # Split features and labels
    X_train = train_df.drop("Label", axis=1).values
    y_train = train_df["Label"].values
    X_test = test_df.drop("Label", axis=1).values
    y_test = test_df["Label"].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Print class distribution
    print("\nClass distribution in training set:")
    class_counts = np.bincount(y_train)
    for i, count in enumerate(class_counts):
        print(f"{class_names[i]}: {count}")

    # Calculate class weights for balanced training
    class_weights = torch.FloatTensor(len(class_counts) / class_counts).to(device)

    # Create datasets and dataloaders
    train_dataset = IoTDataset(X_train, y_train)
    test_dataset = IoTDataset(X_test, y_test)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 100  # Increased epochs since we have early stopping
    train_losses = []
    val_losses = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Training loop
    print("\nTraining MLP model...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(
                    device
                ), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < early_stopping.val_loss_min:
            print(
                f"Validation loss decreased ({early_stopping.val_loss_min:.6f} --> {avg_val_loss:.6f}). Saving model..."
            )
            torch.save(model.state_dict(), "saved_models/mlp_model.pth")
            early_stopping.val_loss_min = avg_val_loss

        early_stopping(avg_val_loss, model)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Load best model
    model.load_state_dict(torch.load("saved_models/mlp_model.pth"))

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(
                device
            )
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(conf_matrix, class_names)

    # Save scaler
    joblib.dump(scaler, "saved_models/mlp_scaler.pkl")

    # Model statistics
    print("\nModel Statistics:")
    print(f"Input dimensions: {input_dim}")
    print(f"Hidden layer sizes: [128, 64]")
    print(f"Output dimensions: {num_classes}")
    print(f"Dropout rate: 0.1")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: 0.001")
    print(f"Early stopping patience: 7")
    print(f"Device used: {device}")


if __name__ == "__main__":
    train_mlp()
