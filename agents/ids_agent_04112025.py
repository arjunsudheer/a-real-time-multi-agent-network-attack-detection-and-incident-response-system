import os
import sys
import torch
import joblib
import numpy as np
import requests
import re
import json
import webbrowser
import shutil
from pathlib import Path
from datetime import datetime, date
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket
from jinja2 import Environment, FileSystemLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import transformers
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Optional
import time

# Get the directory containing this file
AGENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = AGENT_DIR.parent

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# Add the current directory to Python path
sys.path.append(str(PROJECT_ROOT))

from classifiers.iot_torch_mlp import MLPClassifier

# Set environment variables to disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set clean_up_tokenization_spaces globally
transformers.tokenization_utils_base.CLEAN_UP_TOKENIZATION_SPACES = True


class CVE(BaseModel):
    """Schema for a CVE vulnerability"""

    id: str = Field(description="CVE ID in format CVE-YYYY-NNNNN")
    url: str = Field(description="URL to the NVD database entry")
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW)")
    score: str = Field(description="CVSS score")
    description: str = Field(description="Description of the vulnerability")


class SecurityProduct(BaseModel):
    """Schema for a security product recommendation"""

    name: str = Field(
        description="Name of the security product or vendor",
        min_length=2,
        max_length=100,
    )
    type: str = Field(
        description="Category of security product (e.g., Network Security, Endpoint Protection, SIEM, etc.)",
        min_length=3,
        max_length=100,
    )
    description: str = Field(
        description="Detailed description of the product's key security capabilities",
        min_length=50,
        max_length=500,
    )
    relevance_score: float = Field(
        description="Score (0-10) indicating relevance to the security need",
        ge=0.0,
        le=10.0,
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Palo Alto Networks Next-Generation Firewall",
                    "type": "Network Security",
                    "description": "Enterprise-grade firewall with advanced threat prevention, intrusion detection, and application-level security features.",
                    "relevance_score": 9.5,
                }
            ]
        }


class ArxivPaper(BaseModel):
    """Schema for an arXiv research paper"""

    title: str = Field(description="Title of the paper")
    authors: str = Field(description="Authors of the paper")
    summary: str = Field(description="Summary/abstract of the paper")
    url: str = Field(description="URL to the paper on arXiv")
    published: str = Field(description="Publication date")


class SecurityAnalysis(BaseModel):
    """Schema for complete security analysis output"""

    cves: List[CVE] = Field(description="List of related CVE vulnerabilities")
    products: List[SecurityProduct] = Field(
        description="List of recommended security products"
    )
    research: List[ArxivPaper] = Field(description="List of relevant research papers")


class IDSAgent:
    def __init__(self):
        # Set up directories
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.templates_dir = self.base_dir / "templates"
        self.reports_dir = self.base_dir / "reports"
        self.static_dir = self.reports_dir / "static"

        # Create necessary directories
        self.reports_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)), autoescape=True
        )

        # Add custom filters
        def format_datetime(value):
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S UTC")
            return value

        self.jinja_env.filters["format_datetime"] = format_datetime

        # Initialize other components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_sample = None
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
        self.arxiv_retriever = ArxivRetriever(
            load_max_docs=5,
            doc_content_chars_max=None,  # Don't limit content
            max_retries=3,
        )

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
                func=lambda x: self.arxiv_retriever.get_relevant_documents(x),
                description="Retrieve relevant academic papers about IoT security from arXiv",
            ),
            Tool(
                name="CVESearch",
                func=self.search_cve,
                description="Search for CVE vulnerabilities related to IoT devices or protocols. Input should be a search term like 'IoT camera vulnerability' or 'MQTT protocol'",
            ),
            Tool(
                name="ProductRecommendations",
                func=self.get_product_recommendations,
                description="Search for industry product solutions and tools to address specific security needs. Input should be a security need like 'IoT network monitoring' or 'IoT firewall solutions'",
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
            4. Known CVE vulnerabilities related to detected threats
            5. Latest research and knowledge about detected threats
            6. Recommendations for network security
            7. Specific product solutions that could help address identified threats

            Remember to:
            - Compare results from multiple classifiers (at least 3)
            - Consider the strengths of each classifier type
            - Look for patterns in the predictions
            - Search for CVE vulnerabilities related to detected threats
            - Search for latest information about detected threats
            - Reference relevant academic research when available
            - Suggest specific security products and solutions when threats are identified
            - Provide clear explanations of your reasoning

            To use a tool, please use the following format:
            Thought: I need to analyze this sample with multiple classifiers
            Action: the action to take, should be one of [{tool_names}]
            Action Input: (leave empty for classifiers, provide search terms for Search, CVESearch, ProductRecommendations, and KnowledgeRetriever)
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

    def search_cve(self, query: str) -> List[CVE]:
        """Search for CVE vulnerabilities using the NVD API with structured output"""
        try:
            # NVD API endpoint
            base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0/"

            # Get API key from environment variable
            api_key = os.getenv("NVD_API_KEY")

            # Clean and enhance query
            clean_query = re.sub(r"[^\w\s-]", "", query)
            if clean_query.lower() == "icmp flood":
                search_terms = ["ICMP flood", "ping flood", "ICMP DoS"]
            elif clean_query.lower() == "benign":
                return []  # No vulnerabilities for benign traffic
            else:
                search_terms = [clean_query, "DoS", "network attack"]

            enhanced_query = " OR ".join(f'"{term}"' for term in search_terms)

            # Parameters for the API request
            params = {"keywordSearch": enhanced_query, "resultsPerPage": 10}

            # Headers including API key if available
            headers = {"User-Agent": "IDS-Analysis-Tool/1.0"}
            if api_key:
                headers["apiKey"] = api_key

            print(f"\nSearching CVEs with query: {enhanced_query}")

            # Make the request with timeout
            response = requests.get(
                base_url, params=params, headers=headers, timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("totalResults", 0) == 0:
                    # Try alternative search
                    return self._get_alternative_vulnerability_data(query)

                cves = []
                for vuln in data.get("vulnerabilities", [])[:5]:
                    cve_data = vuln.get("cve", {})

                    # Get CVE ID
                    cve_id = cve_data.get("id")
                    if not cve_id or not re.match(r"^CVE-\d{4}-\d{4,7}$", cve_id):
                        continue

                    # Get metrics
                    metrics = cve_data.get("metrics", {})
                    cvss_data = None

                    # Try different CVSS versions
                    for metric_type in [
                        "cvssMetricV31",
                        "cvssMetricV30",
                        "cvssMetricV2",
                    ]:
                        if metric_type in metrics and metrics[metric_type]:
                            cvss_data = metrics[metric_type][0].get("cvssData", {})
                            break

                    if cvss_data:
                        severity = cvss_data.get("baseSeverity", "UNKNOWN")
                        score = str(cvss_data.get("baseScore", "N/A"))
                    else:
                        severity = "UNKNOWN"
                        score = "N/A"

                    # Get description
                    descriptions = cve_data.get("descriptions", [])
                    description = next(
                        (
                            desc["value"]
                            for desc in descriptions
                            if desc.get("lang") == "en"
                        ),
                        "No description available",
                    )

                    cves.append(
                        CVE(
                            id=cve_id,
                            url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                            severity=severity,
                            score=score,
                            description=description,
                        )
                    )

                return cves

            else:
                print(f"NVD API returned status code: {response.status_code}")
                return self._get_alternative_vulnerability_data(query)

        except Exception as e:
            print(f"Error accessing NVD API: {str(e)}")
            return self._get_alternative_vulnerability_data(query)

    def _get_alternative_vulnerability_data(self, query: str) -> List[CVE]:
        """Get vulnerability data from alternative sources when NVD API is unavailable"""
        try:
            # Enhance search query based on traffic type
            clean_query = re.sub(r"[^\w\s-]", "", query)
            if clean_query.lower() == "icmp flood":
                search_terms = [
                    "ICMP flood vulnerability",
                    "ping flood CVE",
                    "ICMP DoS attack CVE",
                ]
            elif clean_query.lower() == "benign":
                return []
            else:
                search_terms = [
                    f"{clean_query} CVE vulnerability",
                    f"{clean_query} security exploit",
                ]

            all_results = []
            for search_term in search_terms:
                try:
                    results = self.search_tool.run(
                        f"{search_term} site:cve.mitre.org OR site:nvd.nist.gov"
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"Error searching term '{search_term}': {str(e)}")

            # Combine all results
            combined_results = "\n".join(all_results)

            # Extract CVE IDs and information
            cves = []
            cve_pattern = r"CVE-\d{4}-\d{4,7}"
            severity_indicators = {
                "CRITICAL": ["critical", "severe", "rce", "remote code execution"],
                "HIGH": ["high", "dangerous", "dos", "denial of service"],
                "MEDIUM": ["medium", "moderate", "xss", "csrf"],
                "LOW": ["low", "minor", "local"],
            }

            matches = re.finditer(cve_pattern, combined_results)
            seen_cves = set()

            for match in matches:
                cve_id = match.group()
                if cve_id in seen_cves:
                    continue

                # Find the surrounding context (200 characters before and after)
                start = max(0, match.start() - 200)
                end = min(len(combined_results), match.end() + 200)
                context = combined_results[start:end]

                # Determine severity based on context
                severity = "MEDIUM"  # Default severity
                context_lower = context.lower()
                for sev, indicators in severity_indicators.items():
                    if any(ind in context_lower for ind in indicators):
                        severity = sev
                        break

                # Extract a meaningful description
                sentences = re.split(r"[.!?]+", context)
                description = ""
                for sentence in sentences:
                    if cve_id in sentence:
                        description = sentence.strip()
                        # Add the next sentence for more context if available
                        next_idx = sentences.index(sentence) + 1
                        if next_idx < len(sentences):
                            description += ". " + sentences[next_idx].strip()
                        break

                if not description:
                    description = f"Vulnerability related to {query} attacks"

                # Assign a score based on severity
                score_ranges = {
                    "CRITICAL": "9.0-10.0",
                    "HIGH": "7.0-8.9",
                    "MEDIUM": "4.0-6.9",
                    "LOW": "0.1-3.9",
                }

                cves.append(
                    CVE(
                        id=cve_id,
                        url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        severity=severity,
                        score=score_ranges.get(severity, "N/A"),
                        description=description,
                    )
                )
                seen_cves.add(cve_id)

                if len(cves) >= 5:  # Limit to 5 results
                    break

            return cves

        except Exception as e:
            print(f"Error in alternative vulnerability search: {str(e)}")
            return []

    def get_product_recommendations(self, search_query: str) -> List[SecurityProduct]:
        """Get security product recommendations with structured output"""
        try:
            print(f"\nSearching for security products related to: {search_query}")

            # Enhance search query
            search_terms = [
                f"{search_query} enterprise security solutions reviews",
                f"top rated {search_query} security products comparison",
                f"best enterprise {search_query} security vendors",
            ]

            all_results = []
            for term in search_terms:
                try:
                    results = self.search_tool.run(term)
                    if results:
                        all_results.append(results)
                        print(f"Found results for: {term}")
                except Exception as e:
                    print(f"Error searching '{term}': {str(e)}")
                    continue

            if not all_results:
                print("No search results found")
                return []

            # Combine results
            combined_results = "\n\n".join(all_results)

            # Extract product information using LLM
            prompt = f"""Extract the top security products/solutions from this text. Focus on enterprise-grade security solutions.

Search Results:
{combined_results}

Return a JSON array of the top 5 most relevant products. Each product should have:
- name: The product name (must be real)
- type: Category (Network Security, IDS/IPS, Firewall, etc.)
- description: Key features and capabilities
- relevance_score: 0-10 based on relevance to {search_query}

Format:
[
    {{
        "name": "Product Name",
        "type": "Product Category",
        "description": "Description",
        "relevance_score": 9.5
    }}
]"""

            # Get LLM response
            response = self.llm.invoke(prompt)

            try:
                # Parse response
                if isinstance(response, str):
                    # Clean up the response
                    cleaned_response = re.sub(
                        r"^```json\s*|\s*```$", "", response.strip()
                    )
                    products_data = json.loads(cleaned_response)
                else:
                    products_data = response

                # Convert to SecurityProduct objects
                products = []
                for product_data in products_data:
                    try:
                        product = SecurityProduct(
                            name=product_data["name"],
                            type=product_data["type"],
                            description=product_data["description"],
                            relevance_score=float(product_data["relevance_score"]),
                        )
                        products.append(product)
                        print(f"Added product: {product.name}")
                    except Exception as e:
                        print(f"Error parsing product: {str(e)}")
                        continue

                return sorted(products, key=lambda x: x.relevance_score, reverse=True)

            except Exception as e:
                print(f"Error parsing products: {str(e)}")
                return []

        except Exception as e:
            print(f"Error in product recommendations: {str(e)}")
            return []

    def _categorize_security_product(self, description):
        """Categorize security product based on description"""
        categories = {
            "Network Security": ["firewall", "ids", "ips", "network", "traffic"],
            "Endpoint Protection": ["endpoint", "antivirus", "edr", "xdr"],
            "Cloud Security": ["cloud", "saas", "container"],
            "Identity & Access": ["identity", "access", "authentication", "iam"],
            "Threat Intelligence": ["threat", "intelligence", "detection"],
            "Vulnerability Management": ["vulnerability", "scanner", "assessment"],
            "SIEM & Analytics": ["siem", "log", "analytics", "monitoring"],
            "IoT Security": ["iot", "device", "embedded"],
        }

        desc_lower = description.lower()
        matches = []

        for category, keywords in categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                matches.append(category)

        return " & ".join(matches[:2]) if matches else "General Security"

    def _calculate_product_relevance(self, product, security_need):
        """Calculate product relevance score based on security need"""
        relevance = 0
        need_lower = security_need.lower()
        desc_lower = product["description"].lower()

        # Check if product name or type matches security need
        if any(word in product["name"].lower() for word in need_lower.split()):
            relevance += 3

        if any(word in product["type"].lower() for word in need_lower.split()):
            relevance += 2

        # Check for security need keywords in description
        if any(word in desc_lower for word in need_lower.split()):
            relevance += 2

        # Bonus for enterprise/professional terms
        enterprise_terms = [
            "enterprise",
            "professional",
            "business",
            "corporate",
            "industry",
        ]
        if any(term in desc_lower for term in enterprise_terms):
            relevance += 1

        return relevance

    def _copy_static_files(self):
        """Copy static assets to the reports directory"""
        # Create CSS file with Tailwind utilities
        tailwind_css = """
        @import 'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css';
        @import 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
        }
        """

        css_file = self.static_dir / "style.css"
        css_file.write_text(tailwind_css)

    def generate_report(self, samples_df):
        """Generate HTML report for all analyzed samples"""
        # Clear existing reports
        if self.reports_dir.exists():
            shutil.rmtree(self.reports_dir)
        self.reports_dir.mkdir()
        self.static_dir.mkdir()

        # Copy static files
        self._copy_static_files()

        all_samples = []

        for i, sample in enumerate(samples_df.iterrows(), 1):
            # Store the current sample
            self.current_sample = sample[1].to_frame().T

            # Get basic information
            sample_info = {
                "id": i,
                "source_port": int(sample[1]["Src Port"]),
                "source_ip": int(sample[1]["Src IP"]),
                "dest_port": int(sample[1]["Dst Port"]),
                "dest_ip": int(sample[1]["Dst IP"]),
                "protocol": int(sample[1]["Protocol"]),
                "detail_url": f"sample_{i}.html",
                "traffic_type": self.label_encoder.inverse_transform(
                    [int(sample[1]["Label"])]
                )[0],
            }

            # Get classifier results
            classifier_results = self.analyze_with_majority_vote()
            sample_info.update(self._parse_classifier_results(classifier_results))

            # Get related research
            traffic_type = sample_info["traffic_type"]
            try:
                # Construct ArXiv query
                if traffic_type.lower() == "icmp flood":
                    query = 'cat:cs.CR AND (ti:"DDoS" OR ti:"denial of service" OR abs:"ICMP flood")'
                elif traffic_type.lower() == "benign":
                    query = 'cat:cs.CR AND (ti:"network traffic classification" OR ti:"intrusion detection")'
                else:
                    query = (
                        f'cat:cs.CR AND (ti:"{traffic_type}" OR abs:"{traffic_type}")'
                    )

                print(f"\nSearching ArXiv with query: {query}")
                try:
                    arxiv_results = self.arxiv_retriever.get_relevant_documents(query)
                    print(f"Found {len(arxiv_results)} papers")

                    if not arxiv_results:
                        # Try fallback query
                        fallback_query = 'cat:cs.CR AND (ti:"network security" OR ti:"intrusion detection")'
                        print(f"\nNo results, trying fallback query: {fallback_query}")
                        arxiv_results = self.arxiv_retriever.get_relevant_documents(
                            fallback_query
                        )
                        print(f"Found {len(arxiv_results)} papers with fallback query")
                except Exception as e:
                    print(f"Error in ArXiv search: {str(e)}")
                    arxiv_results = []

                # Parse and store the results
                sample_info["arxiv_articles"] = self._parse_arxiv_results(arxiv_results)
                print(
                    f"Total papers found and parsed: {len(sample_info['arxiv_articles'])}"
                )

            except Exception as e:
                print(f"Error retrieving research papers: {str(e)}")
                sample_info["arxiv_articles"] = []

            # Get CVE information
            cve_results = self.search_cve(traffic_type)
            sample_info["cves"] = self._parse_cve_results(cve_results)

            # Get product recommendations
            try:
                if traffic_type.lower() == "icmp flood":
                    product_query = "DDoS and ICMP flood protection"
                elif traffic_type.lower() == "benign":
                    product_query = "network monitoring and intrusion detection"
                else:
                    product_query = f"{traffic_type} attack prevention"

                print(f"\nGetting product recommendations for: {product_query}")
                products = self.get_product_recommendations(product_query)
                sample_info["products"] = products
                print(f"Found {len(products)} product recommendations")

            except Exception as e:
                print(f"Error getting product recommendations: {str(e)}")
                sample_info["products"] = []

            all_samples.append(sample_info)

            # Generate individual sample report
            self._generate_sample_report(sample_info)

        # Generate index page
        self._generate_index_page(all_samples)

        return str(self.reports_dir)

    def _parse_classifier_results(self, results):
        """Parse classifier results into structured data"""
        classifiers = []
        majority_vote = {}

        for line in results.split("\n"):
            if "🔹" in line and ":" in line:
                # Parse individual classifier results
                classifier_name = line.split("🔹")[1].split(":")[0].strip()
                prediction = line.split(":")[1].split("(")[0].strip()
                confidence = None
                if "(confidence:" in line:
                    confidence = (
                        float(line.split("confidence:")[1].split(")")[0].strip()) * 100
                    )

                classifiers.append(
                    {
                        "name": classifier_name,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                )
            elif "🎯 Final Prediction:" in line:
                majority_vote["prediction"] = line.split(":")[1].strip()
            elif "📊 Agreement Ratio:" in line:
                majority_vote["agreement_ratio"] = (
                    float(line.split(":")[1].strip()) * 100
                )

        return {"classifiers": classifiers, "majority_vote": majority_vote}

    def _parse_arxiv_results(self, documents):
        """Parse ArXiv search results into structured paper data"""
        papers = []

        try:
            print(f"\nProcessing {len(documents)} ArXiv documents")

            for doc in documents:
                try:
                    # ArxivRetriever gives us structured metadata
                    metadata = doc.metadata
                    print(f"Available fields in metadata: {list(metadata.keys())}")

                    paper = {
                        "title": metadata["Title"],
                        "authors": metadata["Authors"],
                        "arxiv_id": metadata["Entry ID"].split("/")[-1],
                        "url": metadata["Entry ID"],
                        "published": metadata["Published"].isoformat(),
                        "summary": doc.page_content,
                    }

                    papers.append(paper)
                    print(f"- Successfully parsed paper: {paper['title'][:100]}")

                except Exception as e:
                    print(f"- Error parsing paper: {str(e)}")
                    continue

            return papers

        except Exception as e:
            print(f"Error in _parse_arxiv_results: {str(e)}")
            return []

    def _parse_cve_results(self, vulnerabilities):
        """Format CVE results in a structured way"""
        # If the vulnerabilities are already CVE objects, just return them
        if vulnerabilities and isinstance(vulnerabilities[0], CVE):
            return vulnerabilities[:5]

        # Otherwise parse from raw data
        cves = []
        for vuln in vulnerabilities[:5]:
            try:
                cve = vuln["cve"]
                cve_id = cve["id"]

                # Validate CVE ID format
                if not re.match(r"^CVE-\d{4}-\d{4,7}$", cve_id):
                    continue

                # Get English description
                description = next(
                    (
                        desc["value"]
                        for desc in cve.get("descriptions", [])
                        if desc.get("lang") == "en"
                    ),
                    None,
                )

                if not description:
                    continue

                # Get metrics
                metrics = cve.get("metrics", {})
                severity = "N/A"
                score = "N/A"

                for metric_type in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if metric_type in metrics:
                        metric_data = metrics[metric_type][0].get("cvssData", {})
                        severity = metric_data.get("baseSeverity", severity)
                        score = str(metric_data.get("baseScore", score))
                        break

                cves.append(
                    CVE(
                        id=cve_id,
                        url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                        severity=severity,
                        score=score,
                        description=description,
                    )
                )
            except (KeyError, IndexError):
                continue

        return cves

    def _parse_product_results(self, products):
        """Parse product recommendations into structured data"""
        # If products are already SecurityProduct objects, just return them
        if products and isinstance(products[0], SecurityProduct):
            return products[:5]

        # Otherwise parse from raw data
        return [
            SecurityProduct(
                name=product["name"],
                type=product["type"],
                description=product["description"],
                relevance_score=self._calculate_product_relevance(product),
            )
            for product in products[:5]
        ]

    def _generate_sample_report(self, sample_info):
        """Generate HTML report for individual sample"""
        template = self.jinja_env.get_template("traffic_analysis.html")
        sample_info["current_time"] = datetime.utcnow()
        output = template.render(**sample_info)

        output_path = self.reports_dir / sample_info["detail_url"]
        output_path.write_text(output)

    def _generate_index_page(self, all_samples):
        """Generate index page with all samples"""
        template = self.jinja_env.get_template("index.html")
        output = template.render(samples=all_samples, current_time=datetime.utcnow())

        output_path = self.reports_dir / "index.html"
        output_path.write_text(output)

    def _find_available_port(self, start_port=8000):
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError("Could not find an available port")

    def _start_http_server(self, port):
        """Start HTTP server in a separate thread"""
        # Use absolute path of reports directory
        os.chdir(str(self.reports_dir.absolute()))

        server = HTTPServer(("", port), SimpleHTTPRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True  # Thread will exit when main program exits
        server_thread.start()

        return server

    def serve_reports(self):
        """Serve the reports directory and open in browser"""
        # Find an available port
        port = self._find_available_port()

        # Print debug information
        print(f"\nServing reports from: {self.reports_dir.absolute()}")
        print(f"Contents of reports directory:")
        for item in self.reports_dir.iterdir():
            print(f"  - {item.name}")

        # Start the server
        server = self._start_http_server(port)

        # Open the browser
        url = f"http://localhost:{port}/index.html"
        print(f"\nStarting report server at {url}")
        webbrowser.open(url)

        try:
            print("\nPress Ctrl+C to stop the server...")
            while True:
                pass
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
            server.server_close()

    def generate_analysis(self, cves, products, papers) -> SecurityAnalysis:
        """Generate complete security analysis with structured output"""
        return SecurityAnalysis(
            cves=self._parse_cve_results(cves),
            products=self._parse_product_results(products),
            research=self._parse_arxiv_results(papers),
        )


if __name__ == "__main__":
    # Initialize the agent
    agent = IDSAgent()

    # Test arXiv search and product recommendations directly
    print("\n=== Testing ArXiv Search ===")
    try:
        # Test with a known traffic type
        test_query = "DDoS attack"
        print(f"\nTesting arXiv search with query: {test_query}")
        arxiv_results = agent.arxiv_retriever.get_relevant_documents(
            f'all:"{test_query}" AND cat:cs.CR'
        )
        print(f"Found {len(arxiv_results) if arxiv_results else 0} papers")
        if arxiv_results:
            for i, doc in enumerate(arxiv_results[:2], 1):
                print(f"\nPaper {i}:")
                print(f"Title: {doc.metadata.get('title', 'No title')}")
                print(f"Authors: {doc.metadata.get('authors', 'No authors')}")
    except Exception as e:
        print(f"Error in arXiv test: {str(e)}")

    print("\n=== Testing Product Recommendations ===")
    try:
        test_need = "DDoS attack prevention"
        print(f"\nTesting product recommendations for: {test_need}")
        products = agent.get_product_recommendations(test_need)
        print(f"\nFound {len(products)} products")
        for i, product in enumerate(products, 1):
            print(f"\nProduct {i}:")
            print(f"Name: {product.name}")
            print(f"Type: {product.type}")
            print(f"Description: {product.description}")
            print(f"Relevance: {product.relevance_score}")
    except Exception as e:
        print(f"Error in product recommendations test: {str(e)}")

    # Load and sample test data
    print("\n=== Testing Full Analysis ===")
    import pandas as pd

    test_df = pd.read_csv("test.csv")
    samples = test_df.sample(n=5, random_state=42)  # Take 5 random samples

    print(f"\nAnalyzing {len(samples)} random samples...")

    # Generate HTML report
    report_path = agent.generate_report(samples)
    print(f"\nReport generated at: {report_path}")

    # Serve and open the reports
    agent.serve_reports()
