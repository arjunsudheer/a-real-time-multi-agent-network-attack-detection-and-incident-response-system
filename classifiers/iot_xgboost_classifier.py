import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    conf_matrix, class_names, output_file="visualizations/xgboost/confusion_matrix.png"
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
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_xgboost():
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

    # Calculate class weights
    total_samples = len(y_train)
    class_weights = dict(
        zip(
            range(len(class_names)),
            [
                total_samples / (len(class_names) * sum(y_train == c))
                for c in range(len(class_names))
            ],
        )
    )
    print("\nClass weights:")
    for i, weight in class_weights.items():
        print(f"{class_names[i]}: {weight:.2f}")

    # Initialize XGBoost classifier with simpler parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(class_names),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    # Prepare sample weights based on class weights
    sample_weights = np.array([class_weights[y] for y in y_train])

    # Train the model
    print("\nTraining XGBoost model...")
    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )
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
    joblib.dump(model, "saved_models/xgboost_model.pkl")

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
    plt.title("Top 15 Most Important Features")
    plt.tight_layout()
    plt.savefig("visualizations/xgboost/feature_importance.png")
    plt.close()

    # Print some additional metrics
    print("\nPer-class Probability Distribution:")
    class_probs = y_pred_proba.mean(axis=0)
    for i, (name, prob) in enumerate(zip(class_names, class_probs)):
        print(f"{name}: {prob:.4f}")


if __name__ == "__main__":
    train_xgboost()
