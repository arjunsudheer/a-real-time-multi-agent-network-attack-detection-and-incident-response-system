import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import json
import torch
from evaluate_binary_classifier import evaluate_binary_classifier
import sys
from pathlib import Path

# Add project root to Python path to import MLP
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from classifiers.iot_torch_mlp import MLPClassifier


def load_models():
    """Load pre-trained models"""
    print("Loading pre-trained models...")
    models = {}

    # Load PyTorch MLP model
    try:
        input_dim = 78  # Number of features
        num_classes = 9  # Number of classes including benign
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mlp_model = MLPClassifier(input_dim, num_classes).to(device)
        mlp_model.load_state_dict(
            torch.load("saved_models/mlp_model.pth", map_location=device)
        )
        mlp_model.eval()
        models["mlp"] = mlp_model
        print("Successfully loaded MLP model")
    except Exception as e:
        print(f"Error loading MLP model: {str(e)}")

    # Load other models
    try:
        models["rf"] = joblib.load("saved_models/random_forest_model.pkl")
        print("Successfully loaded Random Forest model")
    except Exception as e:
        print(f"Error loading Random Forest model: {str(e)}")

    try:
        models["xgb"] = joblib.load("saved_models/xgboost_model.pkl")
        print("Successfully loaded XGBoost model")
    except Exception as e:
        print(f"Error loading XGBoost model: {str(e)}")

    try:
        models["knn"] = joblib.load("saved_models/knn_model.pkl")
        print("Successfully loaded KNN model")
    except Exception as e:
        print(f"Error loading KNN model: {str(e)}")

    return models


def majority_vote(predictions):
    """Implement majority voting from multiple classifiers"""
    return np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions
    )


def evaluate_ensemble():
    # First, run binary classifier evaluation to get pre-screened data
    X_test, y_true_original, binary_pred = evaluate_binary_classifier()

    # Load pre-trained models
    print("\nLoading pre-trained models...")
    try:
        models = load_models()
    except Exception as e:
        print(f"Error: Could not load pre-trained models. {str(e)}")
        return

    # Get malicious samples from test set based on binary classifier
    malicious_mask_test = binary_pred == 1
    X_test_malicious = X_test[malicious_mask_test]
    y_true_malicious = y_true_original[malicious_mask_test]

    print(f"\nNumber of samples classified as malicious: {len(X_test_malicious)}")
    print(f"True label distribution in malicious samples:")
    for label, count in sorted(zip(*np.unique(y_true_malicious, return_counts=True))):
        print(f"Label {label}: {count}")

    # Get predictions from each model
    print("\nMaking predictions with each model...")
    predictions = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model in models.items():
        if name == "mlp":
            # Handle PyTorch MLP model
            X_tensor = torch.FloatTensor(X_test_malicious.values).to(device)
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probabilities, dim=1).cpu().numpy()
        else:
            # Handle scikit-learn models
            pred = model.predict(X_test_malicious)

        predictions[name] = pred

    # Calculate majority vote
    print("Calculating ensemble predictions...")
    ensemble_pred = majority_vote(np.array(list(predictions.values())))

    # Calculate metrics for each model and ensemble
    results = {"individual_models": {}, "ensemble": {}}

    # Get original class names
    label_encoder = joblib.load("label_encoder.pkl")
    class_names = label_encoder.classes_

    # Individual model metrics
    for name, pred in predictions.items():
        accuracy = accuracy_score(y_true_malicious, pred)
        precision = precision_score(y_true_malicious, pred, average="weighted")
        recall = recall_score(y_true_malicious, pred, average="weighted")
        f1 = f1_score(y_true_malicious, pred, average="weighted")

        results["individual_models"][name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    # Ensemble metrics
    accuracy = accuracy_score(y_true_malicious, ensemble_pred)
    precision = precision_score(y_true_malicious, ensemble_pred, average="weighted")
    recall = recall_score(y_true_malicious, ensemble_pred, average="weighted")
    f1 = f1_score(y_true_malicious, ensemble_pred, average="weighted")

    results["ensemble"] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    # Save results
    with open("experiments/results/ensemble_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print results
    print("\nEnsemble Classification Results:")
    print("\nIndividual Model Performance:")
    for name, metrics in results["individual_models"].items():
        print(f"\n{name.upper()} Classifier:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    print("\nEnsemble Voting Performance:")
    print(f"Accuracy: {results['ensemble']['accuracy']:.4f}")
    print(f"Precision: {results['ensemble']['precision']:.4f}")
    print(f"Recall: {results['ensemble']['recall']:.4f}")
    print(f"F1 Score: {results['ensemble']['f1_score']:.4f}")

    # Plot confusion matrix for ensemble predictions
    conf_matrix = confusion_matrix(y_true_malicious, ensemble_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Post-classification Label Confusion Matrix on Pre-screened Malicious Samples")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("visualizations/post_classification_label_confusion_matrix.png")
    plt.close()

    # Additional analysis: Per-class metrics for ensemble
    print("\nPer-class Performance (Ensemble):")
    per_class_recall = recall_score(y_true_malicious, ensemble_pred, average=None)
    per_class_precision = precision_score(y_true_malicious, ensemble_pred, average=None)
    per_class_f1 = f1_score(y_true_malicious, ensemble_pred, average=None)

    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"Precision: {per_class_precision[i]:.4f}")
        print(f"Recall: {per_class_recall[i]:.4f}")
        print(f"F1 Score: {per_class_f1[i]:.4f}")


if __name__ == "__main__":
    evaluate_ensemble()
