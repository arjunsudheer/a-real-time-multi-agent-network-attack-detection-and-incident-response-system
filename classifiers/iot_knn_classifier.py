import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    conf_matrix, class_names, output_file="visualizations/knn/confusion_matrix.png"
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
    plt.title("KNN Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_knn():
    # Load the preprocessed data
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Load label encoder to get class names
    label_encoder = joblib.load("label_encoder.pkl")
    class_names = label_encoder.classes_

    # Take a stratified sample of 5% of the training data
    train_sample = train_df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * 0.05), random_state=42)
    )

    # Take a stratified sample of 5% of the test data
    test_sample = test_df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * 0.05), random_state=42)
    )

    # Split features and labels
    X_train = train_sample.drop("Label", axis=1)
    y_train = train_sample["Label"].astype(int)
    X_test = test_sample.drop("Label", axis=1)
    y_test = test_sample["Label"].astype(int)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Print class distribution
    print("\nClass distribution in training set:")
    for i, count in enumerate(np.bincount(y_train)):
        print(f"{class_names[i]}: {count}")

    # Scale the features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize KNN with square root of n as k
    k = int(np.sqrt(len(X_train)))
    print(f"\nUsing k={k} neighbors")

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",  # Weight points by inverse of distance
        metric="euclidean",
        n_jobs=-1,  # Use all CPU cores
    )

    # Train the model
    print("\nTraining KNN model...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_scaled)

    # Print classification report with original class names
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, class_names)

    # Save the model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, "saved_models/knn_model.pkl")
    joblib.dump(scaler, "saved_models/knn_scaler.pkl")

    # Calculate feature importance using variance between classes
    print("\nCalculating feature importance...")
    importances = []
    y_train_np = y_train.to_numpy()

    for i in range(X_train_scaled.shape[1]):
        feature_values = X_train_scaled[:, i]
        class_means = [
            np.mean(feature_values[y_train_np == c]) for c in range(len(class_names))
        ]
        importance = np.var(
            class_means
        )  # Higher variance between class means = more important feature
        importances.append(importance)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": importances}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance.head(15))
    plt.title("Top 15 Most Important Features (KNN)")
    plt.tight_layout()
    plt.savefig("visualizations/knn/feature_importance.png")
    plt.close()

    # Print some model statistics
    print("\nModel Statistics:")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of neighbors (k): {k}")

    # Calculate and print average distance to k nearest neighbors
    distances, _ = model.kneighbors(X_test_scaled)
    print(f"Average distance to nearest neighbors: {np.mean(distances):.4f}")
    print(f"Min distance to nearest neighbors: {np.min(distances):.4f}")
    print(f"Max distance to nearest neighbors: {np.max(distances):.4f}")


if __name__ == "__main__":
    train_knn()
