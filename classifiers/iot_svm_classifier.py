import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    conf_matrix, class_names, output_file="visualizations/svm/confusion_matrix.png"
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
    plt.title("SVM Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def train_svm():
    # Load the preprocessed data
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Load label encoder to get class names
    label_encoder = joblib.load("label_encoder.pkl")
    class_names = label_encoder.classes_

    # Take a stratified sample of 10% of the training data
    train_sample = train_df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * 0.1), random_state=42)
    )

    # Take a stratified sample of 10% of the test data
    test_sample = test_df.groupby("Label", group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * 0.1), random_state=42)
    )

    # Split features and labels
    X_train = train_sample.drop("Label", axis=1)
    y_train = train_sample["Label"].astype(int)
    X_test = test_sample.drop("Label", axis=1)
    y_test = test_sample["Label"].astype(int)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Initialize LinearSVC with minimal parameters
    model = LinearSVC(
        dual=False,  # Faster for n_samples > n_features
        class_weight=class_weights,
        random_state=42,
        max_iter=1000,
        tol=1e-3,
    )

    # Train the model
    print("\nTraining SVM model...")
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
    joblib.dump(model, "saved_models/svm_model.pkl")
    joblib.dump(scaler, "saved_models/svm_scaler.pkl")

    # Feature importance analysis (using absolute values of coefficients)
    importances = np.abs(model.coef_).mean(axis=0)
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": importances}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance.head(15))
    plt.title("Top 15 Most Important Features (SVM Coefficients)")
    plt.tight_layout()
    plt.savefig("visualizations/svm/feature_importance.png")
    plt.close()


if __name__ == "__main__":
    train_svm()
