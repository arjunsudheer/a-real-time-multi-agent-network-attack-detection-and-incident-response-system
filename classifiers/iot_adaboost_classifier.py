import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    conf_matrix, class_names, output_file="visualizations/adaboost/confusion_matrix.png"
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
    plt.title("AdaBoost Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_learning_curve(
    estimator_errors, output_file="visualizations/adaboost/learning_curve.png"
):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(estimator_errors) + 1), estimator_errors, "b-")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Error Rate")
    plt.title("AdaBoost Learning Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_adaboost():
    # Load the preprocessed data
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Load label encoder to get class names
    label_encoder = joblib.load("label_encoder.pkl")
    class_names = label_encoder.classes_

    # Split features and labels
    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"].astype(int)
    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"].astype(int)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Print class distribution
    print("\nClass distribution in training set:")
    class_counts = np.bincount(y_train)
    for i, count in enumerate(class_counts):
        print(f"{class_names[i]}: {count}")

    # Initialize base estimator (weak learner)
    base_estimator = DecisionTreeClassifier(
        max_depth=3,  # Shallow trees as weak learners
        min_samples_split=50,  # Minimum samples required to split
        min_samples_leaf=20,  # Minimum samples in leaf nodes
        random_state=42,
    )

    # Initialize AdaBoost
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,  # Number of weak learners
        learning_rate=0.5,  # Contribution of each classifier
        algorithm="SAMME",  # Use discrete predictions
        random_state=42,
    )

    # Train the model
    print("\nTraining AdaBoost model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Print classification report with original class names
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, class_names)

    # Save the model
    print("\nSaving model...")
    joblib.dump(model, "saved_models/adaboost_model.pkl")

    # Plot learning curve using estimator errors
    if hasattr(model, "estimator_errors_"):
        plot_learning_curve(model.estimator_errors_)

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
    plt.title("Top 15 Most Important Features (AdaBoost)")
    plt.tight_layout()
    plt.savefig("visualizations/adaboost/feature_importance.png")
    plt.close()

    # Print model statistics
    print("\nModel Statistics:")
    print(f"Number of estimators: {model.n_estimators}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Number of features: {X_train.shape[1]}")

    # Calculate and print per-class prediction probabilities
    print("\nPer-class Probability Distribution:")
    class_probs = y_pred_proba.mean(axis=0)
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        print(f"{name}: {prob:.4f}")

    # Print estimator weights if available
    if hasattr(model, "estimator_weights_"):
        weights = model.estimator_weights_
        print("\nEstimator Weight Statistics:")
        print(f"Mean weight: {np.mean(weights):.4f}")
        print(f"Min weight: {np.min(weights):.4f}")
        print(f"Max weight: {np.max(weights):.4f}")

        # Plot estimator weights distribution
        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=30)
        plt.xlabel("Estimator Weight")
        plt.ylabel("Count")
        plt.title("Distribution of Estimator Weights")
        plt.savefig("visualizations/adaboost/weights_distribution.png")
        plt.close()


if __name__ == "__main__":
    train_adaboost()
