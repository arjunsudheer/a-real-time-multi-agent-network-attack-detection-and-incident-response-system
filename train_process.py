import pandas as pd
import joblib
from pathlib import Path

from agents.labeling_agent import LabelingAgent
from preprocessing.data_cleaning import preprocess_dataset
from attack_detection_pipeline.pre_detection import PreDetection
from attack_detection_pipeline.post_classification import PostClassification


if __name__ == "__main__":
    # ACI IOT Dataset 2023
    df = pd.read_csv(
        "datasets/aci_iot_network_dataset_2023/original_dataset/ACI-IoT-2023.csv"
    )
    (
        X_train_preprocessed,
        y_train_preprocessed,
        X_test_preprocessed,
        y_test_preprocessed,
        _,
        y_train_original,
        X_test_original,
        y_test_original,
    ) = preprocess_dataset(
        df,
        label_column="Label",
        parent_directory=Path("datasets/aci_iot_network_dataset_2023"),
    )

    # Attack detection pipeline
    # Pre-detection training and evaluation
    pre_detection = PreDetection(
        X_train_preprocessed,
        y_train_original,
        dataset_directory=Path("datasets/aci_iot_network_dataset_2023"),
    )
    pre_detection.save_classification_metrics(X_test_preprocessed, y_test_original)

    # Pre-detection malicious sample filtering
    (
        malicious_preprocessed_samples_df,
        malicious_original_samples_df,
        malicious_indices,
    ) = pre_detection.filter_malicious_samples(
        preprocessed_samples=X_test_preprocessed, original_samples=X_test_original
    )
    # Also filter the labels based on the malicious indices
    filtered_preprocessed_labels = y_test_preprocessed[malicious_indices]
    filtered_original_labels = y_test_original.to_numpy()[malicious_indices]

    # Post-classification training and evaluation
    post_classification = PostClassification(
        X_train_preprocessed,
        y_train_preprocessed,
        dataset_directory=Path("datasets/aci_iot_network_dataset_2023"),
    )
    post_classification.save_classification_metrics(
        X_test_preprocessed,
        y_test_preprocessed,
        unique_labels=joblib.load(
            Path("datasets/aci_iot_network_dataset_2023") / "label_binarizer.pkl"
        ).classes_,
    )

    # Post-classification low agreement sample filtering
    (
        predictions,
        low_agreement_preprocessed_samples_df,
        low_agreement_original_samples_df,
        low_agreement_indices,
    ) = post_classification.filter_low_agreement_samples(
        preprocessed_malicious_samples=malicious_preprocessed_samples_df,
        original_malicious_samples=malicious_original_samples_df,
    )
    # Also filter the labels based on the low agreement indices
    filtered_preprocessed_labels = filtered_preprocessed_labels[low_agreement_indices]
    filtered_original_labels = filtered_original_labels[low_agreement_indices]

    # Prompt labeling agent to resolve the final class prediction for low_agreement samples
    la = LabelingAgent(
        dataset_directory=Path("datasets/aci_iot_network_dataset_2023"),
        label_column="Label",
    )
    la.build_long_term_memory(
        samples=low_agreement_original_samples_df, labels=filtered_original_labels
    )
