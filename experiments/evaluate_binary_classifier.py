import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import json


def evaluate_binary_classifier():
    # Load the test data
    print("Loading test data...")
    test_df = pd.read_csv("test.csv")

    # Convert labels to binary (0 for Benign, 1 for all others)
    y_true = (test_df["Label"] != 0).astype(int)
    X_test = test_df.drop("Label", axis=1)

    # Load the trained binary model
    print("Loading binary classifier model...")
    model = joblib.load("saved_models/binary_random_forest_model.pkl")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Create results dictionary
    results = {
        "binary_classifier_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
    }

    # Save metrics to JSON file
    with open("experiments/results/binary_classifier_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print metrics
    print("\nBinary Classifier Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    plt.title("Binary Classifier Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("visualizations/binary_random_forest_classifier_confusion_matrix.png")
    plt.close()

    # Return predictions and true labels for further analysis
    return X_test, test_df["Label"], y_pred


if __name__ == "__main__":
    evaluate_binary_classifier()
