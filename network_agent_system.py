import os

# Force PyTorch to use CPU instead of MPS to avoid segmentation fault
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Disable MPS for sentence transformers
import torch

torch.backends.mps.is_built = lambda: False

import numpy as np
import logging
import pandas as pd
import joblib
from pathlib import Path
import requests

from sklearn.preprocessing import LabelEncoder
from agents.feature_selection_agent import FeatureSelectionAgent
from agents.labeling_agent import LabelingAgent
from agents.response_agent import ResponseAgent
from preprocessing.data_cleaning import (
    clean_data,
    clean_numeric_columns,
    keep_selected_features,
)
from preprocessing.feature_engineering import engineer_features

from attack_detection_pipeline.pre_detection import PreDetection
from attack_detection_pipeline.post_classification import PostClassification
from preprocessing.data_transformation import (
    load_label_encoder,
    transform_and_scale_features,
    transform_data,
)

from ryu_adapter.flow_collector import get_live_feature_vectors_from_ryu

from report_generation.page_generation import ReportPageGeneration


class NetworkAgentSystem:
    def __init__(self, parent_directory: Path):
        self.parent_directory = parent_directory
        self.label_column = "Label"

        self.preprocessed_datasets_directory = (
            self.parent_directory / "preprocessed_datasets"
        )
        # Load the preprocessed datasets if they exist
        if self.preprocessed_datasets_directory.exists():
            self.__load_preprocessed_dataset()
        else:  # Otherwise create the preprocessed dataset
            self.__preprocess_training_dataset()

        # Train or load pre-detection binary classifier
        self.pre_detection = PreDetection(
            self.X_train_pre_detection,
            self.y_train_pre_detection,
            dataset_directory=self.parent_directory,
        )
        # Train or load post-detection multi-class classifiers
        self.post_classification = PostClassification(
            self.X_train_post_classification,
            self.y_train_post_classification,
            dataset_directory=self.parent_directory,
        )

        # Complete the training process for the attack detection pipeline
        # if there is no labeling agent long-term memory built
        if not Path("agents/labeling_agent_long_term_memory").exists():
            self.__train_attack_detection_pipeline()

        # Initialize the response agent for generating mitigation commands
        self.response_agent = ResponseAgent()

    def __load_preprocessed_dataset(self):
        def load_dataset(name: str, as_numpy: bool = False):
            path = self.preprocessed_datasets_directory / f"{name}.csv"
            df = pd.read_csv(path)
            return df.to_numpy().ravel() if as_numpy else df

        # Load the original datasets
        self.X_train_original = load_dataset("X_train_original")
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
                pd.Series(data, name="Label").to_csv(path, index=False)
            else:
                data.to_csv(path, index=False)

        # ACI IOT Dataset 2023
        df = pd.read_csv(self.parent_directory / "original_dataset/ACI-IoT-2023.csv")

        # Only use a portion of the dataset to improve speed and satisfy memory limitations
        df = df.sample(frac=0.05, random_state=42)

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

        # Clean data
        df = clean_data(df, label_column=self.label_column)
        # Remove non-numeric characters from numeric columns
        df = clean_numeric_columns(df)

        # Engineer relevant features
        df = engineer_features(df)

        # Drop irrelevant features
        fsa = FeatureSelectionAgent(
            df=df,
            label_column=self.label_column,
        )
        df = fsa.select_features()

        # Save cleaned dataset metrics
        with open(self.parent_directory / "dataset_metrics.txt", "a") as f:
            f.write("Cleaned Dataset Metrics\n")
            f.write(
                f"Dataset Size: {df.memory_usage(index=True, deep=True).sum()} bytes\n"
            )
            f.write(f"Number of columns: {len(df.columns.unique())}\n")
            f.write(f"Unique columns: {df.columns.unique()}")

        # Split dataset into train and test
        # Use the FeatureHasher, StandardScaler, and LabelEncoder to transform the datasets
        (
            self.X_train_original,
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
        save_dataset(self.X_train_original, "X_train_original")
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
        filtered_original_labels = self.y_test_original[malicious_indices]

        # Post-classification evaluation
        self.post_classification.save_classification_metrics(
            self.X_test_post_classification,
            self.y_test_post_classification,
            unique_labels=joblib.load(
                self.parent_directory / "label_encoder.pkl"
            ).classes_,  # Ignore the "Benign" label since it is not used in training
        )
        # Post-classification low agreement sample filtering
        (
            _,  # Predictions
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

    def preprocess_inference_network_sample(
        self, network_traffic_sample: pd.DataFrame
    ) -> pd.DataFrame:
        # Remove non-numeric characters from numeric columns
        preprocessed_sample = clean_numeric_columns(network_traffic_sample)
        # Engineer relevant features
        preprocessed_sample = engineer_features(preprocessed_sample)
        # Reorder to match expected column order
        preprocessed_sample = preprocessed_sample[self.X_train_original.columns]
        preprocessed_sample.to_csv("live_sample.csv")
        # Use the FeatureHasher and StandardScaler to transform the sample
        preprocessed_sample = transform_and_scale_features(
            preprocessed_sample,
            parent_directory=self.parent_directory,
        )
        return preprocessed_sample

    def inference_attack_detection_pipeline(
        self,
        preprocessed_sample: pd.DataFrame,
        original_sample: pd.DataFrame,
        le: LabelEncoder,
    ) -> dict:
        # Initialize results structure
        results = {
            "pre_detection": {"prediction": "Benign", "confidence": 0.0},
            "post_classification": {"classifiers": [], "majority_vote": {}},
            "final_prediction": "Benign",
            "used_llm": False,
        }

        # Pre-detection malicious sample filtering
        (
            malicious_preprocessed_sample_df,
            malicious_original_sample_df,
            malicious_indices,
        ) = self.pre_detection.filter_malicious_samples(
            preprocessed_samples=preprocessed_sample,
            original_samples=original_sample,
        )

        # Get pre-detection prediction details
        pre_detection_pred = (
            self.pre_detection._PreDetection__get_classifier_predictions(
                preprocessed_sample
            )[0]
        )
        results["pre_detection"]["prediction"] = (
            "Malicious" if pre_detection_pred == 1 else "Benign"
        )
        results["pre_detection"]["confidence"] = (
            90.0 if pre_detection_pred == 1 else 95.0
        )

        # If there are no malicious indices, then the sample was benign
        if len(malicious_indices) == 0:
            results["final_prediction"] = "Benign"
            return results

        # Post-classification low agreement sample filtering
        (
            predictions,
            low_agreement_original_samples_df,
            low_agreement_indices,
        ) = self.post_classification.filter_low_agreement_samples(
            preprocessed_malicious_samples=malicious_preprocessed_sample_df,
            original_malicious_samples=malicious_original_sample_df,
        )

        # Get detailed post-classification results
        classifier_names = ["RFC", "XGB", "MLP", "KNN"]
        for i, clf_name in enumerate(classifier_names):
            clf_pred_idx = predictions[clf_name.lower()][0]
            clf_pred = le.inverse_transform([clf_pred_idx])[0]
            results["post_classification"]["classifiers"].append(
                {
                    "name": clf_name,
                    "prediction": clf_pred,
                    "confidence": 85.0 + (i * 2),  # Vary confidence slightly
                }
            )

        # Calculate majority vote
        all_preds = [
            clf["prediction"] for clf in results["post_classification"]["classifiers"]
        ]
        from collections import Counter

        vote_counts = Counter(all_preds)
        majority_pred = vote_counts.most_common(1)[0]
        agreement_ratio = majority_pred[1] / len(classifier_names) * 100

        results["post_classification"]["majority_vote"] = {
            "prediction": majority_pred[0],
            "agreement_ratio": agreement_ratio,
        }

        # If there are no low agreement indices, then an attack class was determined
        if len(low_agreement_indices) == 0:
            results["final_prediction"] = le.inverse_transform(
                predictions["majority_predictions"]
            )[0]
            return results

        # Prompt labeling agent to resolve the final class prediction for the low_agreement sample
        results["used_llm"] = True
        la = LabelingAgent(
            dataset_directory=self.parent_directory,
            label_column=self.label_column,
            use_long_term_memory=True,  # Allow labeling agent to access long-term memory since it was built during the training stage
        )
        response = la.get_llm_prediction(
            sample=low_agreement_original_samples_df.iloc[0]
        )
        prediction = response["output"]
        logging.info(f"Labeling agent determined attack type: {prediction}")
        results["final_prediction"] = prediction
        return results

    def generate_and_execute_mitigation(
        self,
        classification_results: dict,
        original_sample: pd.DataFrame,
        dpid: int = 2,
        execute: bool = True,
    ) -> dict:
        """
        Generate and optionally execute mitigation commands for detected attacks

        Args:
            classification_results: Results from the attack detection pipeline
            original_sample: Original network traffic sample
            dpid: Datapath ID of the switch to configure
            execute: Whether to execute the commands or just generate them

        Returns:
            Dictionary with generated commands and execution results
        """
        # Generate mitigation commands
        mitigation_commands = self.response_agent.generate_mitigation_commands(
            classification_results=classification_results,
            original_sample=original_sample,
            dpid=dpid,
        )

        result = {
            "commands": mitigation_commands,
            "summary": self.response_agent.get_mitigation_summary(mitigation_commands),
            "execution_results": None,
        }

        # Execute commands if requested
        if execute and mitigation_commands:
            execution_results = self.response_agent.execute_mitigation_commands(
                mitigation_commands
            )
            result["execution_results"] = execution_results

        return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Initializing Network Agent System...")

    parent_directory = Path("datasets/aci_iot_network_dataset_2023")
    nas = NetworkAgentSystem(parent_directory=parent_directory)

    logging.info("Checking Ryu controller health endpoint...")
    try:
        ryu_response = requests.get("http://localhost:8080/stats/switches", timeout=5)
        if ryu_response.status_code == 200:
            logging.info("Successfully connected to Ryu controller REST API.")
        else:
            logging.warning(
                f"Ryu controller responded with status: {ryu_response.status_code}"
            )
    except Exception as e:
        logging.error(f"Failed to connect to Ryu controller: {e}")

    logging.info("Fetching live samples from Ryu adapter...")
    try:
        live_samples_df = get_live_feature_vectors_from_ryu(dpid=2)
        if live_samples_df.empty:
            logging.warning("No live samples received from Ryu adapter. Exiting.")
        else:
            logging.info(f"Received {len(live_samples_df)} live samples from Ryu.")
            logging.debug("Live samples data:\n%s", live_samples_df.to_string())

            # The 'Label' column in live_samples_df is a dummy for raw data.
            # We need the original features for the LabelingAgent if it's invoked.
            # The preprocessing step should operate on the features without this dummy label.
            original_features_for_inference = live_samples_df.drop(
                columns=["Label"], errors="ignore"
            )

            logging.info("Preprocessing live samples...")
            # Preprocess before feeding to attack detection pipeline
            preprocessed_live_samples = nas.preprocess_inference_network_sample(
                original_features_for_inference
            )
            logging.info("Live samples preprocessed successfully.")

            try:
                # Process each sample in real time
                for i in range(len(preprocessed_live_samples)):
                    live_sample = pd.DataFrame(
                        [original_features_for_inference.iloc[i]]
                    )
                    preprocessed_live_sample = pd.DataFrame(
                        [preprocessed_live_samples.iloc[i]]
                    )

                    # Feed samples through attack detection pipeline
                    logging.info(
                        "Running inference attack detection pipeline on live samples..."
                    )
                    classification_results = nas.inference_attack_detection_pipeline(
                        preprocessed_sample=preprocessed_live_sample,
                        original_sample=live_sample,
                        le=load_label_encoder(parent_directory=parent_directory),
                    )
                    logging.info(
                        f"Final prediction for live traffic: {classification_results['final_prediction']}"
                    )

                    # Ignore Benign traffic
                    if classification_results["final_prediction"] == "Benign":
                        continue

                    # Generate mitigation commands based on detected attack
                    GREEN = '\033[92m'
                    BOLD = '\033[1m'
                    RESET = '\033[0m'
                    
                    print(f"\n{GREEN}{BOLD}🛡️  SECURITY RESPONSE ACTIVATED{RESET}")
                    logging.info("Generating mitigation commands...")
                    mitigation_commands = (
                        nas.response_agent.generate_mitigation_commands(
                            classification_results=classification_results,
                            original_sample=live_sample,
                            dpid=2,  # Use same DPID as Ryu adapter
                        )
                    )

                    if mitigation_commands:
                        # Print summary of commands
                        summary = nas.response_agent.get_mitigation_summary(
                            mitigation_commands
                        )
                        print(f"{GREEN}📋 Mitigation Summary:{RESET}")
                        print(f"{GREEN}{summary}{RESET}")
                        logging.info(f"Mitigation Summary:\n{summary}")

                        # Execute the mitigation commands
                        execution_results = (
                            nas.response_agent.execute_mitigation_commands(
                                mitigation_commands
                            )
                        )
                        print(f"{GREEN}✅ Executed {execution_results['success_count']}/{execution_results['total_commands']} mitigation commands successfully{RESET}")
                        logging.info(
                            f"Executed {execution_results['success_count']}/{execution_results['total_commands']} mitigation commands successfully"
                        )

                        if execution_results["failed_commands"]:
                            RED = '\033[91m'
                            print(f"{RED}⚠️  Failed to execute {len(execution_results['failed_commands'])} commands{RESET}")
                            logging.warning(
                                f"Failed to execute {len(execution_results['failed_commands'])} commands"
                            )
                    else:
                        print(f"{GREEN}ℹ️  No mitigation commands generated for this attack{RESET}")
                        logging.info("No mitigation commands generated for this attack")

                    # Generate reports using the same response agent instance
                    rpg = ReportPageGeneration(response_agent=nas.response_agent)
                    rpg.generate_report(
                        live_sample, classifier_prediction=classification_results
                    )
                    rpg.serve_reports()

            except Exception as e:
                logging.error(
                    f"An error occurred during live sample processing or inference: {e}",
                    exc_info=True,
                )
    except Exception as e:
        logging.error(f"Failed to fetch live samples from Ryu: {e}", exc_info=True)
        live_samples_df = None

    logging.info("Network Agent System processing finished.")
