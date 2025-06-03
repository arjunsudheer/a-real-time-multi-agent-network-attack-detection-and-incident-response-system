import sys
import numpy as np
import logging # Added for logging
import pandas as pd
import joblib
from pathlib import Path

from agents.feature_selection_agent import FeatureSelectionAgent
from agents.labeling_agent import LabelingAgent
from preprocessing.data_cleaning import (
    clean_data,
    clean_numeric_columns,
    keep_selected_features,
)
from attack_detection_pipeline.pre_detection import PreDetection
from attack_detection_pipeline.post_classification import PostClassification
from preprocessing.data_transformation import (
    transform_and_scale_features,
    transform_data,
)
from ryu_adapter.flow_collector import get_live_feature_vectors_from_ryu # Added for Ryu integration


class NetworkAgentSystem:
    def __init__(self):
        self.parent_directory = Path("datasets/aci_iot_network_dataset_2023")
        self.label_column = "Label"

        self.preprocessed_datasets_directory = (
            self.parent_directory / "preprocessed_datasets"
        )
        # Load the preprocessed datasets if they exist
        # Assume training process has completed if the preprocessed datasets exist
        if self.preprocessed_datasets_directory.exists():
            self.__load_dataset()
        else:  # Otherwise create the preprocessed dataset
            self.__preprocess_training_dataset()
            # Complete the training process for all other components
            self.__train_attack_detection_pipeline()
            self.__train_response_system()
            self.__train_recommendation_agent()

    def __load_dataset(self):
        def load_dataset(name: str, as_numpy: bool = False):
            path = self.preprocessed_datasets_directory / f"{name}.csv"
            df = pd.read_csv(path, header=None)
            return df.to_numpy().ravel() if as_numpy else df

        # Load the original datasets
        self.y_train_original = load_dataset("y_train_original", as_numpy=True)
        self.X_test_original = load_dataset("X_test_original")
        self.y_test_original = load_dataset("y_test_original", as_numpy=True)

        # Load the pre-detection datasets
        self.X_train_pre_detection = load_dataset("X_train_pre_detection")
        self.y_train_pre_detection = load_dataset(
            "y_train_pre_detection", as_numpy=True
        )
        self.X_test_pre_detection = load_dataset("X_test_pre_detection")
        self.y_test_pre_detection = load_dataset("y_test_pre_detection", as_numpy=True)

        # Load the post-classification datasets
        self.X_train_post_classification = load_dataset("X_train_post_classification")
        self.y_train_post_classification = load_dataset(
            "y_train_post_classification", as_numpy=True
        )
        self.X_test_post_classification = load_dataset("X_test_post_classification")
        self.y_test_post_classification = load_dataset(
            "y_test_post_classification", as_numpy=True
        )

    def __preprocess_training_dataset(self) -> list[pd.DataFrame | np.ndarray]:
        def save_dataset(data: pd.DataFrame | np.ndarray, name: str):
            path = self.preprocessed_datasets_directory / f"{name}.csv"
            if isinstance(data, np.ndarray):
                pd.Series(data).to_csv(path, index=True)
            else:
                data.to_csv(path, index=True)

        # ACI IOT Dataset 2023
        df = pd.read_csv(self.parent_directory / "original_dataset/ACI-IoT-2023.csv")

        # Only use a portion of the dataset to improve speed and satisfy memory limitations
        df = df.sample(frac=0.2, random_state=42)

        # Save initial dataset metrics
        with open(self.parent_directory / "dataset_metrics.txt", "w") as f:
            f.write("Initial Dataset Metrics\n")
            f.write(
                f"Dataset Size: {df.memory_usage(index=True, deep=True).sum()} bytes\n"
            )
            f.write(f"Number of columns: {len(df.columns.unique())}\n")
            f.write(f"Unique columns: {df.columns.unique()}\n\n")

        # Only keep the features that exist during inference
        features_during_inference = [
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Flow Duration",
            "Flow Bytes/s",
            "Flow Packets/s",
        ]
        df = keep_selected_features(
            df,
            features_to_keep=features_during_inference,
            label_column=self.label_column,
        )

        # Drop irrelevant features
        fsa = FeatureSelectionAgent(
            df=df,
            label_column=self.label_column,
            dataset_name=self.parent_directory.name,
        )
        df = fsa.select_features()

        # Clean data
        df = clean_data(df, label_column=self.label_column)
        # Remove non-numeric characters from numeric columns
        df = clean_numeric_columns(df)

        # Save cleaned dataset metrics
        with open(self.parent_directory / "dataset_metrics.txt", "a") as f:
            f.write("Cleaned Dataset Metrics\n")
            f.write(
                f"Dataset Size: {df.memory_usage(index=True, deep=True).sum()} bytes\n"
            )
            f.write(f"Number of columns: {len(df.columns.unique())}\n")
            f.write(f"Unique columns: {df.columns.unique()}")

        # Split dataset into train and test
        # Use the FeatureHasher, StandardScaler, and LabelBinarizer to transform the datasets
        (
            _,  # X_train original
            self.y_train_original,
            self.X_test_original,
            self.y_test_original,
            self.X_train_pre_detection,
            self.y_train_pre_detection,
            self.X_test_pre_detection,
            self.y_test_pre_detection,
            self.X_train_post_classification,
            self.y_train_post_classification,
            self.X_test_post_classification,
            self.y_test_post_classification,
        ) = transform_data(
            df,
            label_column=self.label_column,
            parent_directory=self.parent_directory,
        )

        self.preprocessed_datasets_directory.mkdir()
        # Save the original datasets
        save_dataset(self.y_train_original, "y_train_original")
        save_dataset(self.X_test_original, "X_test_original")
        save_dataset(self.y_test_original, "y_test_original")

        # Save the pre-detection datasets
        save_dataset(self.X_train_pre_detection, "X_train_pre_detection")
        save_dataset(self.y_train_pre_detection, "y_train_pre_detection")
        save_dataset(self.X_test_pre_detection, "X_test_pre_detection")
        save_dataset(self.y_test_pre_detection, "y_test_pre_detection")

        # Save the post-classification datasets
        save_dataset(
            self.X_train_post_classification,
            "X_train_post_classification",
        )
        save_dataset(
            self.y_train_post_classification,
            "y_train_post_classification",
        )
        save_dataset(
            self.X_test_post_classification,
            "X_test_post_classification",
        )
        save_dataset(
            self.y_test_post_classification,
            "y_test_post_classification",
        )

    def __train_attack_detection_pipeline(self):
        # Train pre-detection binary classifier
        self.pre_detection = PreDetection(
            self.X_train_pre_detection,
            self.y_train_pre_detection,
            dataset_directory=self.parent_directory,
        )
        # Pre-detection evaluation
        self.pre_detection.save_classification_metrics(
            self.X_train_pre_detection, self.y_train_pre_detection
        )
        # Pre-detection malicious sample filtering
        (
            malicious_preprocessed_samples_df,
            malicious_original_samples_df,
            malicious_indices,
        ) = self.pre_detection.filter_malicious_samples(
            preprocessed_samples=self.X_test_pre_detection,
            original_samples=self.X_test_original,
        )
        # Filter the original labels based on the malicious indices to use for building the Labeling Agent's long-term memory
        # filtered_original_labels = self.y_test_original[malicious_indices]
        # Assuming malicious_indices came from X_test_preprocessed or X_test_original
        positions = self.X_test_original.index.get_indexer(malicious_indices)
        filtered_original_labels = self.y_test_original[positions]

        # Train post-detection multi-class classifiers
        self.post_classification = PostClassification(
            self.X_train_post_classification,
            self.y_train_post_classification,
            dataset_directory=self.parent_directory,
        )
        # Post-classification evaluation
        self.post_classification.save_classification_metrics(
            self.X_test_post_classification,
            self.y_test_post_classification,
            unique_labels=joblib.load(
                self.parent_directory / "label_binarizer.pkl"
            ).classes_,  # Ignore the "Benign" label since it is not used in training
        )
        # Post-classification low agreement sample filtering
        (
            predictions,
            low_agreement_original_samples_df,
            low_agreement_indices,
        ) = self.post_classification.filter_low_agreement_samples(
            preprocessed_malicious_samples=malicious_preprocessed_samples_df,
            original_malicious_samples=malicious_original_samples_df,
        )
        # Filter the original labels based on the malicious indices to use for building the Labeling Agent's long-term memory
        filtered_original_labels = filtered_original_labels[low_agreement_indices]

        # Prompt labeling agent to resolve the final class prediction for low_agreement samples
        la = LabelingAgent(
            dataset_directory=self.parent_directory,
            label_column=self.label_column,
        )
        la.build_long_term_memory(
            samples=low_agreement_original_samples_df, labels=filtered_original_labels
        )

    def __train_response_system(self):
        pass

    def __train_recommendation_agent(self):
        pass

    def preprocess_inference_network_sample(self, network_traffic_sample: pd.DataFrame) -> pd.DataFrame:
        # Clean and transform sample
        # Use .copy() to avoid modifying the original DataFrame passed to this function
        processed_sample = network_traffic_sample.copy()
        # Remove non-numeric characters from numeric columns
        processed_sample = clean_numeric_columns(processed_sample)
        # Use the FeatureHasher and StandardScaler to transform the sample
        processed_sample = transform_and_scale_features(
            processed_sample,
            parent_directory=self.parent_directory,
        )
        return processed_sample

    def inference_attack_detection_pipeline(
        self,
        current_preprocessed_samples: pd.DataFrame,
        current_original_samples: pd.DataFrame
    ) -> str:
        """
        Runs the attack detection pipeline on the given live samples.

        Args:
            current_preprocessed_samples (pd.DataFrame): The live samples after preprocessing (hashing/scaling).
            current_original_samples (pd.DataFrame): The live samples in their original feature format (before hashing/scaling).
        Returns:
            str: The predicted attack type or "Benign".
        """
        # Pre-detection malicious sample filtering
        (
            malicious_preprocessed_samples_df,
            malicious_original_samples_df,
            malicious_indices,
        ) = self.pre_detection.filter_malicious_samples(
            preprocessed_samples=current_preprocessed_samples,
            original_samples=current_original_samples,
        )
        # If there are no malicious indices, then the sample was benign
        if len(malicious_indices) == 0:
            if not malicious_preprocessed_samples_df.empty: # if samples were processed but all benign
                logging.info("Pre-detection classified all live samples as Benign.")
            else: # if no samples were passed to filter_malicious_samples (e.g. current_preprocessed_samples was empty)
                logging.warning("No samples available for pre-detection or all were filtered out before pre-detection.")
            return "Benign"

        # Post-classification low agreement sample filtering
        (
            predictions,
            low_agreement_original_samples_df,
            low_agreement_indices,
        ) = self.post_classification.filter_low_agreement_samples(
            preprocessed_malicious_samples=malicious_preprocessed_samples_df,
            original_malicious_samples=malicious_original_samples_df,
        )
        # If there are no low agreement indices, then an attack class was determined
        if len(low_agreement_indices) == 0:
            logging.info(f"Post-classification determined attack type: {predictions[0]}")
            return predictions[0]

        # Prompt labeling agent to resolve the final class prediction for the low_agreement sample
        la = LabelingAgent(
            dataset_directory=self.parent_directory,
            label_column=self.label_column,
        )
        response = la.get_llm_prediction(
            sample=low_agreement_original_samples_df.iloc[0]
        )
        prediction = response["output"]
        logging.info(f"Labeling agent determined attack type: {prediction}")
        return prediction

    def inference_response_system(self):
        pass

    def inference_recommendation_agent(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Initializing Network Agent System...")
    nas = NetworkAgentSystem()

    logging.info("Fetching live samples from Ryu adapter...")
    # Fetch, for example, 5 samples from DPID 1
    live_samples_df = get_live_feature_vectors_from_ryu(num_samples_to_fetch=5, dpid=1)

    if live_samples_df.empty:
        logging.warning("No live samples received from Ryu adapter. Exiting.")
    else:
        logging.info(f"Received {len(live_samples_df)} live samples from Ryu.")
        logging.debug("Live samples data:\n%s", live_samples_df.to_string())

        # The 'Label' column in live_samples_df is a dummy for raw data.
        # We need the original features for the LabelingAgent if it's invoked.
        # The preprocessing step should operate on the features without this dummy label.
        original_features_for_inference = live_samples_df.drop(columns=['Label'], errors='ignore')

        logging.info("Preprocessing live samples...")
        try:
            preprocessed_live_samples_df = nas.preprocess_inference_network_sample(
                original_features_for_inference.copy() # Use .copy() to be safe
            )
            logging.info("Live samples preprocessed successfully.")
            logging.debug("Preprocessed live samples data shape: %s", preprocessed_live_samples_df.shape)

            logging.info("Running inference attack detection pipeline on live samples...")
            prediction = nas.inference_attack_detection_pipeline(
                current_preprocessed_samples=preprocessed_live_samples_df,
                current_original_samples=original_features_for_inference
            )
            logging.info(f"Final prediction for live traffic: {prediction}")

        except Exception as e:
            logging.error(f"An error occurred during live sample processing or inference: {e}", exc_info=True)

    logging.info("Network Agent System processing finished.")
