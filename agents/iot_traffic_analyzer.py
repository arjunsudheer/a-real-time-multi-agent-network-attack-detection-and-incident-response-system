import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifiers.iot_torch_mlp import MLPClassifier  # Import the MLP model class

from dotenv import load_dotenv

load_dotenv()


class IoTTrafficAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        saved_models_dir = Path("saved_models")

        # Load label encoder
        self.label_encoder = joblib.load("label_encoder.pkl")
        print("\nLoading models...")

        # Define valid model types and which ones need scalers
        valid_models = {
            "random_forest",
            "decision_tree",
            "xgboost",
            "svm",
            "knn",
            "adaboost",
            "mlp",
        }
        models_with_scalers = {"svm", "knn", "mlp"}

        # Load all available models and their corresponding scalers if needed
        for model_path in saved_models_dir.glob("*_model.*"):
            # Extract base name (e.g., 'random_forest' from 'random_forest_model')
            model_name = "_".join(
                model_path.stem.split("_")[:-1]
            )  # Join all parts except 'model'
            if model_name not in valid_models:
                print(f"Skipping unknown model type: {model_name}")
                continue

            print(f"\nAttempting to load {model_name} model from {model_path}")

            try:
                # Load the model
                if model_name == "mlp":
                    # Initialize MLP model architecture
                    input_dim = 78  # Number of features
                    num_classes = len(self.label_encoder.classes_)
                    model = MLPClassifier(input_dim, num_classes).to(self.device)
                    # Load the trained weights
                    model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    model.eval()
                else:
                    # Load scikit-learn model
                    model = joblib.load(model_path)

                self.models[model_name] = model
                print(f"Successfully loaded {model_name} model")

                # Load scaler if the model needs it
                if model_name in models_with_scalers:
                    scaler_path = saved_models_dir / f"{model_name}_scaler.pkl"
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
                        print(f"Successfully loaded {model_name} scaler")
                    else:
                        print(f"Warning: Scaler not found for {model_name}")
                        # Remove the model if its required scaler is missing
                        del self.models[model_name]
            except Exception as e:
                print(f"Error loading {model_name} model: {str(e)}")

        print("\nAvailable models:", list(self.models.keys()))

    def analyze_sample(self, sample_df, classifier_type="all"):
        if classifier_type not in self.models and classifier_type != "all":
            raise ValueError(f"Classifier {classifier_type} not available")

        results = {}

        classifiers = (
            [classifier_type] if classifier_type != "all" else self.models.keys()
        )

        for clf_name in classifiers:
            if clf_name not in self.models:
                continue

            # Prepare the features
            X = sample_df.copy()

            # Scale features if needed
            if clf_name in self.scalers:
                X = self.scalers[clf_name].transform(X)

            # Get predictions
            if clf_name == "mlp":
                # PyTorch prediction
                X_tensor = torch.FloatTensor(X).to(self.device)
                with torch.no_grad():
                    outputs = self.models[clf_name](X_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                predictions = predictions.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
            else:
                # Scikit-learn prediction
                predictions = self.models[clf_name].predict(X)
                try:
                    probabilities = self.models[clf_name].predict_proba(X)
                except:
                    # Some models might not support probability predictions
                    probabilities = None

            # Convert numeric predictions to class names
            predicted_classes = self.label_encoder.inverse_transform(predictions)

            # Store results
            results[clf_name] = {
                "predictions": predicted_classes,
                "probabilities": probabilities,
            }

        return results

    def get_available_classifiers(self):
        return list(self.models.keys())


# Initialize the analyzer
analyzer = IoTTrafficAnalyzer()

# Global variable for samples
current_samples = None


# Tool functions that use the classifiers
def analyze_with_xgboost(query: str = "") -> str:
    """Analyze IoT traffic using XGBoost classifier"""
    global current_samples
    results = analyzer.analyze_sample(current_samples.drop("Label", axis=1), "xgboost")
    if "xgboost" not in results:
        return "XGBoost classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(results["xgboost"]["predictions"], results["xgboost"]["probabilities"])
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def analyze_with_svm(query: str = "") -> str:
    """Analyze IoT traffic using SVM classifier"""
    global current_samples
    results = analyzer.analyze_sample(current_samples.drop("Label", axis=1), "svm")
    if "svm" not in results:
        return "SVM classifier not available"

    output = []
    for i, pred in enumerate(results["svm"]["predictions"]):
        output.append(f"Sample {i+1}: {pred}")

    return "\n".join(output)


def analyze_with_decision_tree(query: str = "") -> str:
    """Analyze IoT traffic using Decision Tree classifier"""
    global current_samples
    results = analyzer.analyze_sample(
        current_samples.drop("Label", axis=1), "decision_tree"
    )
    if "decision_tree" not in results:
        return "Decision Tree classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(
            results["decision_tree"]["predictions"],
            results["decision_tree"]["probabilities"],
        )
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def analyze_with_knn(query: str = "") -> str:
    """Analyze IoT traffic using KNN classifier"""
    global current_samples
    results = analyzer.analyze_sample(current_samples.drop("Label", axis=1), "knn")
    if "knn" not in results:
        return "KNN classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(results["knn"]["predictions"], results["knn"]["probabilities"])
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def analyze_with_random_forest(query: str = "") -> str:
    """Analyze IoT traffic using Random Forest classifier"""
    global current_samples
    results = analyzer.analyze_sample(
        current_samples.drop("Label", axis=1), "random_forest"
    )
    if "random_forest" not in results:
        return "Random Forest classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(
            results["random_forest"]["predictions"],
            results["random_forest"]["probabilities"],
        )
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def analyze_with_adaboost(query: str = "") -> str:
    """Analyze IoT traffic using AdaBoost classifier"""
    global current_samples
    results = analyzer.analyze_sample(current_samples.drop("Label", axis=1), "adaboost")
    if "adaboost" not in results:
        return "AdaBoost classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(results["adaboost"]["predictions"], results["adaboost"]["probabilities"])
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def analyze_with_mlp(query: str = "") -> str:
    """Analyze IoT traffic using MLP classifier"""
    global current_samples
    results = analyzer.analyze_sample(current_samples.drop("Label", axis=1), "mlp")
    if "mlp" not in results:
        return "MLP classifier not available"

    output = []
    for i, (pred, probs) in enumerate(
        zip(results["mlp"]["predictions"], results["mlp"]["probabilities"])
    ):
        confidence = probs[np.where(analyzer.label_encoder.classes_ == pred)[0][0]]
        output.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")

    return "\n".join(output)


def get_random_samples(n_samples: int = 5) -> pd.DataFrame:
    """Get random samples from test data"""
    # Load the original scaler used during training
    original_scaler = joblib.load("scaler.pkl")

    # Load normalized test data
    test_df = pd.read_csv("test.csv")
    samples = test_df.sample(n=n_samples, random_state=42)

    # Denormalize features (except Label)
    features_to_denorm = [col for col in samples.columns if col != "Label"]
    samples[features_to_denorm] = original_scaler.inverse_transform(
        samples[features_to_denorm]
    )

    return samples


# Create tools for the agent
tools = [
    Tool(
        name="XGBoost_Analyzer",
        func=analyze_with_xgboost,
        description="Analyzes IoT traffic using XGBoost classifier. Returns predictions and confidence levels.",
    ),
    Tool(
        name="SVM_Analyzer",
        func=analyze_with_svm,
        description="Analyzes IoT traffic using SVM classifier. Returns predictions.",
    ),
    Tool(
        name="DecisionTree_Analyzer",
        func=analyze_with_decision_tree,
        description="Analyzes IoT traffic using Decision Tree classifier. Returns predictions and confidence levels.",
    ),
    Tool(
        name="KNN_Analyzer",
        func=analyze_with_knn,
        description="Analyzes IoT traffic using KNN classifier. Returns predictions and confidence levels.",
    ),
    Tool(
        name="RandomForest_Analyzer",
        func=analyze_with_random_forest,
        description="Analyzes IoT traffic using Random Forest classifier. Returns predictions and confidence levels.",
    ),
    Tool(
        name="AdaBoost_Analyzer",
        func=analyze_with_adaboost,
        description="Analyzes IoT traffic using AdaBoost classifier. Returns predictions and confidence levels.",
    ),
    Tool(
        name="MLP_Analyzer",
        func=analyze_with_mlp,
        description="Analyzes IoT traffic using MLP (Neural Network) classifier. Returns predictions and confidence levels.",
    ),
]

# Create React agent prompt
prompt = PromptTemplate.from_template(
    """You are an expert IoT traffic analyzer. Your task is to analyze network traffic patterns using multiple machine learning classifiers and provide insights.

You have access to the following tools:

{tools}

Tool Names: {tool_names}

Use these tools to analyze the traffic and provide detailed insights about:
1. Agreement/disagreement between different classifiers
2. Confidence levels of predictions
3. Potential threats and their severity
4. Recommendations for network security

Remember to:
- Compare results from multiple classifiers
- Consider the strengths of each classifier type
- Look for patterns in the predictions
- Provide clear explanations of your reasoning
- Start with at least 3 different classifiers for comparison
- Pay special attention to any disagreements between classifiers
- Note when confidence levels are particularly high or low

To use a tool, please use the following format:

Thought: I need to analyze this sample with multiple classifiers
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know what to recommend
Final Answer: the final analysis and recommendations

Begin!

Question: {input}

{agent_scratchpad}"""
)

# Initialize Gemini model
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,  # Lower temperature for more focused analysis
)

# Create React agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # More graceful handling of parsing errors
)


def analyze_traffic():
    # Use existing samples
    global current_samples

    # Get actual labels
    actual_labels = [
        analyzer.label_encoder.inverse_transform([label])[0]
        for label in current_samples["Label"].values
    ]

    # Create analysis prompt
    analysis_prompt = f"""Analyze these 5 random traffic samples using the available classifiers. 
    For each sample:
    1. Use multiple classifiers to predict the traffic type
    2. Compare and contrast the predictions
    3. Assess the confidence levels
    4. Provide security recommendations based on the findings
    
    The actual labels for these samples are: {actual_labels}
    
    Please start by using at least 3 different classifiers (e.g., XGBoost, Random Forest, and MLP) 
    to get a comprehensive view of the traffic patterns.
    """

    # Run agent
    response = agent_executor.invoke({"input": analysis_prompt})
    return response


def display_traffic_summary(samples):
    """Display a meaningful summary of traffic samples focusing on key network features"""
    print("\n=== TRAFFIC SAMPLES TO ANALYZE ===")

    for i, (_, sample) in enumerate(samples.iterrows(), 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i} OF {len(samples)}")
        print(f"{'='*60}")

        # Get the traffic type - handle float labels by converting to int first
        label_int = int(sample["Label"])  # Convert float to int
        traffic_type = analyzer.label_encoder.inverse_transform([label_int])[0]

        # Basic Flow Information
        print("\n📊 BASIC FLOW INFORMATION:")
        print(f"🔹 Traffic Type: {traffic_type}")
        print(
            f"🔹 Source → Destination: {sample['Src IP']:.0f}:{int(sample['Src Port'])} → {sample['Dst IP']:.0f}:{int(sample['Dst Port'])}"
        )
        print(f"🔹 Protocol: {int(sample['Protocol'])}")
        print(f"🔹 Flow Duration: {sample['Flow Duration']:.2f} microseconds")

        # Packet Information
        print("\n📦 PACKET INFORMATION:")
        print(
            f"🔹 Forward Packets: {int(sample['Total Fwd Packet'])} (avg size: {sample['Fwd Packet Length Mean']:.2f} bytes)"
        )
        print(
            f"🔹 Backward Packets: {int(sample['Total Bwd packets'])} (avg size: {sample['Bwd Packet Length Mean']:.2f} bytes)"
        )
        print(
            f"🔹 Total Length: Fwd={sample['Total Length of Fwd Packet']:.0f} bytes, Bwd={sample['Total Length of Bwd Packet']:.0f} bytes"
        )

        # Flow Rates
        print("\n📈 FLOW RATES:")
        print(f"🔹 Flow Rate: {sample['Flow Packets/s']:.2f} packets/s")
        print(f"🔹 Bytes Rate: {sample['Flow Bytes/s']:.2f} bytes/s")
        print(f"🔹 Forward Rate: {sample['Fwd Packets/s']:.2f} packets/s")
        print(f"🔹 Backward Rate: {sample['Bwd Packets/s']:.2f} packets/s")

        # TCP Flags
        print("\n🚩 TCP FLAGS:")
        print(f"🔹 FIN: {int(sample['FIN Flag Count'])}")
        print(f"🔹 SYN: {int(sample['SYN Flag Count'])}")
        print(f"🔹 RST: {int(sample['RST Flag Count'])}")
        print(f"🔹 PSH: {int(sample['PSH Flag Count'])}")
        print(f"🔹 ACK: {int(sample['ACK Flag Count'])}")
        print(f"🔹 URG: {int(sample['URG Flag Count'])}")

        # Activity Pattern
        print("\n⚡ ACTIVITY PATTERN:")
        print(f"🔹 Active Mean: {sample['Active Mean']:.2f} microseconds")
        print(f"🔹 Idle Mean: {sample['Idle Mean']:.2f} microseconds")

        print(f"\n{'='*60}\n")

    print("All samples have been displayed.")


if __name__ == "__main__":
    # Get and display traffic samples first
    current_samples = get_random_samples(5)

    # Print column names to see what we're working with
    print("\nAvailable columns:")
    print(current_samples.columns.tolist())

    display_traffic_summary(current_samples)

    print("\nLoading models and starting analysis...")
    result = analyze_traffic()
    print("\nAgent's Analysis:")
    print(result["output"])
