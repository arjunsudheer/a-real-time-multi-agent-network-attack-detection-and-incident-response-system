import os
import sys
import torch
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifiers.iot_torch_mlp import MLPClassifier


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
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize empty FAISS stores for both memories
        self.short_term_memory = FAISS.from_texts(["Initial memory"], self.embeddings)
        self.long_term_memory = FAISS.from_texts(["Initial memory"], self.embeddings)

    def initialize_tools(self):
        """Initialize search and knowledge retrieval tools"""
        # Initialize DuckDuckGo search
        self.search_tool = DuckDuckGoSearchRun()

        # Initialize arXiv retriever
        self.arxiv_retriever = ArxivRetriever()

        # Create classifier tools
        classifier_tools = []
        for model_name in self.models.keys():
            classifier_tools.append(
                Tool(
                    name=f"{model_name.upper()}_Classifier",
                    func=lambda _="", m=model_name: self.analyze_with_classifier(m),
                    description=f"Analyzes IoT traffic using {model_name} classifier. No input needed, just call the tool.",
                )
            )

        # Create search and knowledge tools
        self.tools = classifier_tools + [
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

    def update_memory(self, observation: str, is_short_term: bool = True):
        """Update either short-term or long-term memory"""
        memory = self.short_term_memory if is_short_term else self.long_term_memory
        memory.add_texts([observation])

    def query_memory(self, query: str, is_short_term: bool = True, k: int = 5):
        """Query either short-term or long-term memory"""
        memory = self.short_term_memory if is_short_term else self.long_term_memory
        return memory.similarity_search(query, k=k)

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

    test_df = pd.read_csv("test.csv")
    sample = test_df.sample(n=5, random_state=42)

    result = agent.analyze(sample)
    print("\nAgent's Analysis:")
    print(result)
