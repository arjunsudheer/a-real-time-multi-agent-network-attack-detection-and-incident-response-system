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
    class_names,
    output_file="visualizations/random_forest/confusion_matrix.png",
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
    plt.title("Random Forest Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_random_forest():
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

    # Calculate class weights
    total_samples = len(y_train)
    class_weights = dict(
        zip(
            range(len(class_names)),
            [total_samples / (len(class_names) * count) for count in class_counts],
        )
    )
    print("\nClass weights:")
    for i, weight in class_weights.items():
        print(f"{class_names[i]}: {weight:.2f}")

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
    print("\nTraining Random Forest model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Print classification report with original class names
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_matrix, class_names)

    # Save the model
    print("\nSaving model...")
    joblib.dump(model, "saved_models/random_forest_model.pkl")

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
    plt.title("Top 15 Most Important Features (Random Forest)")
    plt.tight_layout()
    plt.savefig("visualizations/random_forest/feature_importance.png")
    plt.close()

    # Print model statistics
    print("\nModel Statistics:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Maximum depth: {model.max_depth}")
    print(f"Number of features: {X_train.shape[1]}")

    # Calculate and print per-class prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    print("\nPer-class Probability Distribution:")
    class_probs = y_pred_proba.mean(axis=0)
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        print(f"{name}: {prob:.4f}")

    # Calculate and print out-of-bag score if available
    if hasattr(model, "oob_score_"):
        print(f"\nOut-of-bag score: {model.oob_score_:.4f}")


if __name__ == "__main__":
    train_random_forest()
