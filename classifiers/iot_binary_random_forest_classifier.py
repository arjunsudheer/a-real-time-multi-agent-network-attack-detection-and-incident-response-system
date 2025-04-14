import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    conf_matrix,
    output_file="visualizations/binary_random_forest/confusion_matrix.png",
):
    class_names = ["Benign", "Malicious"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Binary Random Forest Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_binary_random_forest():
    # Load the preprocessed data
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Convert multi-class labels to binary (0 for Benign, 1 for all others)
    train_df["Label"] = (train_df["Label"] != 0).astype(int)
    test_df["Label"] = (test_df["Label"] != 0).astype(int)

    # Split features and labels
    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"]
    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Print class distribution
    print("\nClass distribution in training set:")
    class_counts = np.bincount(y_train)
    print(f"Benign: {class_counts[0]}")
    print(f"Malicious: {class_counts[1]}")

    # Calculate class weights
    total_samples = len(y_train)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    print("\nClass weights:")
    print(f"Benign: {class_weights[0]:.2f}")
    print(f"Malicious: {class_weights[1]:.2f}")

    # Initialize Random Forest with balanced parameters
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=15,  # Maximum depth of trees
        min_samples_split=10,  # Minimum samples required to split
        min_samples_leaf=5,  # Minimum samples in leaf nodes
        class_weight=class_weights,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=1,
    )

    # Train the model
    print("\nTraining Binary Random Forest model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix)

    # Save the model
    print("\nSaving model...")
    joblib.dump(model, "saved_models/binary_random_forest_model.pkl")

    # Feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance.head(15))
    plt.title("Top 15 Most Important Features (Binary Random Forest)")
    plt.tight_layout()
    plt.savefig("visualizations/binary_random_forest/feature_importance.png")
    plt.close()

    # Print model statistics
    print("\nModel Statistics:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Maximum depth: {model.max_depth}")
    print(f"Number of features: {X_train.shape[1]}")

    # Calculate and print prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    print("\nOverall Probability Distribution:")
    class_probs = y_pred_proba.mean(axis=0)
    print(f"Average probability for Benign: {class_probs[0]:.4f}")
    print(f"Average probability for Malicious: {class_probs[1]:.4f}")

    # Calculate and print out-of-bag score if available
    if hasattr(model, "oob_score_"):
        print(f"\nOut-of-bag score: {model.oob_score_:.4f}")


if __name__ == "__main__":
    train_binary_random_forest() 