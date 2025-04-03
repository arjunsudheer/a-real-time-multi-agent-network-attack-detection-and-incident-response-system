import os
import sys
import torch
import joblib
import numpy as np
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import transformers

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifiers.iot_torch_mlp import MLPClassifier

# Set environment variables to disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set clean_up_tokenization_spaces globally
transformers.tokenization_utils_base.CLEAN_UP_TOKENIZATION_SPACES = True


class IDSAgent:
    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store the current sample being analyzed
        self.current_sample = None

        # Initialize models and tools
        self.initialize_classifiers()
        self.initialize_memory()
        self.initialize_tools()
        self.initialize_llm()
        self.initialize_agent()

    def initialize_classifiers(self):
        """Initialize all ML classifiers"""
        self.models = {}
        self.scalers = {}
        self.label_encoder = None

        # Load label encoder
        self.label_encoder = joblib.load("label_encoder.pkl")
        print("\nLoading models...")

        saved_models_dir = Path("saved_models")
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

        for model_path in saved_models_dir.glob("*_model.*"):
            model_name = "_".join(model_path.stem.split("_")[:-1])
            if model_name not in valid_models:
                continue

            try:
                if model_name == "mlp":
                    input_dim = 78
                    num_classes = len(self.label_encoder.classes_)
                    model = MLPClassifier(input_dim, num_classes).to(self.device)
                    model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    model.eval()
                else:
                    model = joblib.load(model_path)

                self.models[model_name] = model
                print(f"Successfully loaded {model_name} model")

                if model_name in models_with_scalers:
                    scaler_path = saved_models_dir / f"{model_name}_scaler.pkl"
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
                    else:
                        del self.models[model_name]
            except Exception as e:
                print(f"Error loading {model_name} model: {str(e)}")

    def initialize_memory(self):
        """Initialize FAISS vector store for memory"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"tokenizer_kwargs": {"clean_up_tokenization_spaces": True}},
        )

        # Create memory directory if it doesn't exist
        memory_dir = Path("memory")
        memory_dir.mkdir(exist_ok=True)

        # Paths for memory files
        short_term_path = memory_dir / "short_term_memory"
        long_term_path = memory_dir / "long_term_memory"

        # Initialize or load short-term memory
        if short_term_path.exists():
            self.short_term_memory = FAISS.load_local(
                str(short_term_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.short_term_memory = FAISS.from_texts(
                ["Initial memory"], self.embeddings
            )
            self.short_term_memory.save_local(str(short_term_path))

        # Initialize or load long-term memory
        if long_term_path.exists():
            self.long_term_memory = FAISS.load_local(
                str(long_term_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self.long_term_memory = FAISS.from_texts(
                ["Initial memory"], self.embeddings
            )
            self.long_term_memory.save_local(str(long_term_path))

    def initialize_tools(self):
        """Initialize search and knowledge retrieval tools"""
        # Initialize DuckDuckGo search
        self.search_tool = DuckDuckGoSearchRun()

        # Initialize arXiv retriever
        self.arxiv_retriever = ArxivRetriever()

        # Create unified classifier tool
        self.tools = [
            Tool(
                name="UnifiedClassifier",
                func=lambda _="": self.analyze_with_majority_vote(),
                description="Analyzes IoT traffic using all available classifiers with majority voting while showing individual results. No input needed, just call the tool.",
            ),
            Tool(
                name="Search",
                func=self.search_tool.run,
                description="Search the internet for information about IoT security and threats",
            ),
            Tool(
                name="KnowledgeRetriever",
                func=self.arxiv_retriever.get_relevant_documents,
                description="Retrieve relevant academic papers about IoT security from arXiv",
            ),
        ]

    def initialize_llm(self):
        """Initialize the LLM"""
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

    def initialize_agent(self):
        """Initialize the agent with the React prompt"""
        prompt = PromptTemplate.from_template(
            """You are an expert IoT traffic analyzer and security agent. Your task is to analyze network traffic patterns 
            using multiple machine learning classifiers and provide insights about potential threats.

            You have access to the following tools:
            {tools}

            Use these tools to analyze the traffic and provide detailed insights about:
            1. Agreement/disagreement between different classifiers
            2. Confidence levels of predictions
            3. Potential threats and their severity
            4. Latest research and knowledge about detected threats
            5. Recommendations for network security

            Remember to:
            - Compare results from multiple classifiers (at least 3)
            - Consider the strengths of each classifier type
            - Look for patterns in the predictions
            - Search for latest information about detected threats
            - Reference relevant academic research when available
            - Provide clear explanations of your reasoning

            To use a tool, please use the following format:
            Thought: I need to analyze this sample with multiple classifiers
            Action: the action to take, should be one of [{tool_names}]
            Action Input: (leave empty for classifiers, provide search terms for Search and KnowledgeRetriever)
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know what to recommend
            Final Answer: the final analysis and recommendations

            Begin!

            Question: {input}

            {agent_scratchpad}"""
        )

        # Create React agent
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,  # Increased from default
            max_execution_time=300,  # 5 minutes timeout
            early_stopping_method="generate",
        )

    def analyze_with_classifier(self, classifier_type: str) -> str:
        """Analyze the current sample with a specific classifier"""
        if self.current_sample is None:
            return "No sample available for analysis"

        if classifier_type not in self.models:
            return f"Classifier {classifier_type} not available"

        # Prepare the features
        X = self.current_sample.drop("Label", axis=1)

        # Scale features if needed
        if classifier_type in self.scalers:
            X = self.scalers[classifier_type].transform(X)

        # Get predictions
        if classifier_type == "mlp":
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.models[classifier_type](X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
            predictions = predictions.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        else:
            predictions = self.models[classifier_type].predict(X)
            try:
                probabilities = self.models[classifier_type].predict_proba(X)
            except:
                probabilities = None

        # Convert numeric predictions to class names
        predicted_classes = self.label_encoder.inverse_transform(predictions)

        # Format results
        results = []
        for i, pred in enumerate(predicted_classes):
            if probabilities is not None:
                confidence = probabilities[i][predictions[i]]
                results.append(f"Sample {i+1}: {pred} (confidence: {confidence:.2f})")
            else:
                results.append(f"Sample {i+1}: {pred}")

        return "\n".join(results)

    def analyze_with_majority_vote(self) -> str:
        """Analyze the current sample with all classifiers and perform majority voting"""
        if self.current_sample is None:
            return "No sample available for analysis"

        # Store predictions from all classifiers
        all_predictions = {}
        all_probabilities = {}

        # Get predictions from each classifier
        X = self.current_sample.drop("Label", axis=1)

        for classifier_type, model in self.models.items():
            # Scale features if needed
            X_scaled = X.copy()
            if classifier_type in self.scalers:
                X_scaled = self.scalers[classifier_type].transform(X_scaled)

            # Get predictions
            if classifier_type == "mlp":
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                predictions = predictions.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
            else:
                predictions = model.predict(X_scaled)
                try:
                    probabilities = model.predict_proba(X_scaled)
                except:
                    probabilities = None

            all_predictions[classifier_type] = predictions
            all_probabilities[classifier_type] = probabilities

        # Format results for each sample
        final_results = []
        num_samples = len(self.current_sample)

        for sample_idx in range(num_samples):
            sample_results = [f"\n=== Sample {sample_idx + 1} Analysis ===\n"]

            # Individual classifier results
            sample_results.append("Individual Classifier Predictions:")
            for classifier_type in self.models.keys():
                pred_idx = all_predictions[classifier_type][sample_idx]
                pred_class = self.label_encoder.inverse_transform([pred_idx])[0]

                if all_probabilities[classifier_type] is not None:
                    confidence = all_probabilities[classifier_type][sample_idx][
                        pred_idx
                    ]
                    sample_results.append(
                        f"🔹 {classifier_type.upper()}: {pred_class} (confidence: {confidence:.2f})"
                    )
                else:
                    sample_results.append(f"🔹 {classifier_type.upper()}: {pred_class}")

            # Majority vote
            sample_predictions = [pred[sample_idx] for pred in all_predictions.values()]
            unique_preds, counts = np.unique(sample_predictions, return_counts=True)
            majority_idx = unique_preds[np.argmax(counts)]
            majority_class = self.label_encoder.inverse_transform([majority_idx])[0]
            agreement_ratio = np.max(counts) / len(self.models)

            sample_results.append(f"\nMajority Vote Result:")
            sample_results.append(f"🎯 Final Prediction: {majority_class}")
            sample_results.append(f"📊 Agreement Ratio: {agreement_ratio:.2f}")

            final_results.append("\n".join(sample_results))

        return "\n\n".join(final_results)

    def update_memory(self, observation: str, is_short_term: bool = True):
        """Update either short-term or long-term memory"""
        memory = self.short_term_memory if is_short_term else self.long_term_memory
        memory.add_texts([observation])

        # Save to disk
        memory_path = Path("memory") / (
            "short_term_memory" if is_short_term else "long_term_memory"
        )
        memory.save_local(str(memory_path))

    def query_memory(self, query: str, is_short_term: bool = True, k: int = 5):
        """Query either short-term or long-term memory"""
        memory = self.short_term_memory if is_short_term else self.long_term_memory
        return memory.similarity_search(query, k=k)

    def display_traffic_summary(self, samples):
        """Display a summary of traffic samples"""
        print("\n=== TRAFFIC SAMPLES TO ANALYZE ===\n")

        for i, sample in enumerate(samples.iterrows(), 1):
            print(f"{'='*60}")
            print(f"SAMPLE {i} OF {len(samples)}")
            print(f"{'='*60}\n")

            # Get the traffic type - handle float labels by converting to int first
            label_int = int(sample[1]["Label"])  # Convert float to int
            traffic_type = self.label_encoder.inverse_transform([label_int])[0]

            print("📊 BASIC FLOW INFORMATION:")
            print(f"🔹 Traffic Type: {traffic_type}")
            print(
                f"🔹 Source → Destination: {sample[1]['Src Port']:.0f}:{int(sample[1]['Src IP'])} → {sample[1]['Dst Port']:.0f}:{int(sample[1]['Dst IP'])}"
            )
            print(f"🔹 Protocol: {int(sample[1]['Protocol'])}")
            print(f"🔹 Flow Duration: {sample[1]['Flow Duration']:.2f} microseconds\n")

            print("📦 PACKET INFORMATION:")
            print(
                f"🔹 Forward Packets: {int(sample[1]['Total Fwd Packet'])} (avg size: {sample[1]['Fwd Packet Length Mean']:.2f} bytes)"
            )
            print(
                f"🔹 Backward Packets: {int(sample[1]['Total Bwd packets'])} (avg size: {sample[1]['Bwd Packet Length Mean']:.2f} bytes)"
            )
            print(
                f"🔹 Total Length: Fwd={sample[1]['Total Length of Fwd Packet']:.0f} bytes, Bwd={sample[1]['Total Length of Bwd Packet']:.0f} bytes\n"
            )

            print("📈 FLOW RATES:")
            print(f"🔹 Flow Rate: {sample[1]['Flow Packets/s']:.2f} packets/s")
            print(f"🔹 Bytes Rate: {sample[1]['Flow Bytes/s']:.2f} bytes/s")
            print(f"🔹 Forward Rate: {sample[1]['Fwd Packets/s']:.2f} packets/s")
            print(f"🔹 Backward Rate: {sample[1]['Bwd Packets/s']:.2f} packets/s\n")

            print("🚩 TCP FLAGS:")
            print(f"🔹 FIN: {int(sample[1]['FIN Flag Count'])}")
            print(f"🔹 SYN: {int(sample[1]['SYN Flag Count'])}")
            print(f"🔹 RST: {int(sample[1]['RST Flag Count'])}")
            print(f"🔹 PSH: {int(sample[1]['PSH Flag Count'])}")
            print(f"🔹 ACK: {int(sample[1]['ACK Flag Count'])}")
            print(f"🔹 URG: {int(sample[1]['URG Flag Count'])}\n")

            print("⚡ ACTIVITY PATTERN:")
            print(f"🔹 Active Mean: {sample[1]['Active Mean']:.2f} microseconds")
            print(f"🔹 Idle Mean: {sample[1]['Idle Mean']:.2f} microseconds")

            print(f"\n{'='*60}\n")

        print("All samples have been displayed.")

    def analyze(self, sample_df):
        """Main analysis method"""
        # Store the current sample
        self.current_sample = sample_df

        # Create analysis prompt
        analysis_prompt = f"""Analyze these traffic samples using the available classifiers. 
        For each sample:
        1. Use multiple classifiers to predict the traffic type
        2. Compare and contrast the predictions
        3. Search for information about detected threats
        4. Find relevant academic research
        5. Provide security recommendations
        
        Please start by using at least 3 different classifiers to get a comprehensive view."""

        # Run agent
        response = self.agent_executor.invoke({"input": analysis_prompt})

        # Update memories
        self.update_memory(str(response["output"]), is_short_term=True)
        if "high_severity" in str(response["output"]).lower():
            self.update_memory(str(response["output"]), is_short_term=False)

        return response["output"]


if __name__ == "__main__":
    # Initialize the agent
    agent = IDSAgent()

    # Example usage
    import pandas as pd

    # Load and sample test data
    test_df = pd.read_csv("test.csv")
    sample = test_df.sample(n=5, random_state=42)

    # Display traffic samples first
    agent.display_traffic_summary(sample)

    # Run the analysis
    print("\nStarting detailed analysis...")
    result = agent.analyze(sample)
    print("\nAgent's Analysis:")
    print(result)
