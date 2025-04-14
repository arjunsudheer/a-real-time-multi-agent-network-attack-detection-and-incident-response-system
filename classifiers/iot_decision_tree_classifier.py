import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz


def plot_confusion_matrix(
    conf_matrix,
    class_names,
    output_file="visualizations/decision_tree/confusion_matrix.png",
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
    plt.title("Decision Tree Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def visualize_tree(
    model,
    feature_names,
    class_names,
    output_file="visualizations/decision_tree/tree_structure",
):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=5,  # Limit depth for visualization
    )
    graph = graphviz.Source(dot_data)
    graph.render(output_file, format="pdf")


def train_decision_tree():
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

    # Initialize Decision Tree with interpretable parameters
    model = DecisionTreeClassifier(
        max_depth=10,  # Limit depth for interpretability
        min_samples_split=50,  # Minimum samples required to split
        min_samples_leaf=20,  # Minimum samples in leaf nodes
        class_weight=class_weights,
        random_state=42,
    )

    # Train the model
    print("\nTraining Decision Tree model...")
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
    joblib.dump(model, "saved_models/decision_tree_model.pkl")

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
    plt.title("Top 15 Most Important Features (Decision Tree)")
    plt.tight_layout()
    plt.savefig("visualizations/decision_tree/feature_importance.png")
    plt.close()

    # Visualize the tree structure
    try:
        print("\nGenerating tree visualization...")
        visualize_tree(model, X_train.columns, class_names)
        print("Tree visualization saved as 'decision_tree.pdf'")
    except Exception as e:
        print(f"Could not generate tree visualization: {e}")

    # Print tree statistics
    print("\nTree Statistics:")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    n_nodes = model.tree_.node_count
    print(f"Total number of nodes: {n_nodes}")


if __name__ == "__main__":
    train_decision_tree()
