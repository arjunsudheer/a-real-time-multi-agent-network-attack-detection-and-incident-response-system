from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from pathlib import Path
import os

import numpy as np

from agents.knowledge_source import KnowledgeSource
from preprocessing.data_cleaning import (
    clean_numeric_columns,
    transform_and_scale_features,
    preprocess_single_sample,
)
from classifiers.majority_voting import (
    get_signature_method_classification,
    get_robust_classifier_predictions,
    calculate_majority_classification,
)


class IDSAgent:
    def __init__(self, train_directory: Path) -> None:
        self.train_directory = train_directory

        # Long term memory dataset
        self.ltm_db = KnowledgeSource(Path("datasets/faiss_ltm"))

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_llm(self) -> None:
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

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
        agent = create_react_agent(llm=llm, tools=self.tools, prompt=prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=300,  # 5 minutes timeout
        )

    def __initialize_tools(self, include_ltm: bool = False) -> None:
        # Initialize DuckDuckGo search
        web_search = DuckDuckGoSearchRun()
        # Initialize arXiv retriever
        arxiv_retriever = ArxivRetriever()

        def safe_web_search(query: str) -> str:
            try:
                return web_search.run(query)
            except Exception as e:
                return f"ERROR: Web Search failed: {e}"

        def safe_knowledge_retrieve(query: str) -> str:
            try:
                return arxiv_retriever.run(query)
            except Exception as e:
                return f"ERROR: Academic Literature Search failed: {e}"

        # Create unified classifier tool
        self.tools = [
            Tool(
                name="AcademicLiteratureSearch",
                func=lambda query: safe_knowledge_retrieve(query),
                description="Retrieve relevant academic papers about IoT security from arXiv. Please be specific regarding what you want to find. Mention all features you want to learn more about.",
            ),
            Tool(
                name="WebSearch",
                func=lambda query: safe_web_search(query),
                description="Search the internet for information about IoT security and threats. You need to pass the search query to the tool. Use short queries, do not list many comma-separated values. Keep queries under 20 words.",
            ),
        ]

        if include_ltm:
            self.tools.append(
                Tool(
                    name="AccessPreviousCorrectResponses",
                    func=lambda query: self.ltm_db.retrieve_relevant_knowledge(
                        query=query
                    ),
                    description="Shows the first 100 rows of the dataset. You do not need to pass any inputs, just call the tool.",
                )
            )

    def get_llm_prediction(
        self, sample: np.ndarray, dataset_directory: Path, contains_label: bool = False
    ):
        # Run agent
        response = self.agent_executor.invoke(
            {
                "input": f"""Analyze these traffic samples using the available classifiers. 
                For each sample:
                1. Use multiple classifiers to predict the traffic type
                2. Compare and contrast the predictions
                3. Search for information about detected threats
                4. Find relevant academic research
                5. Provide security recommendations
                
                Please start by using at least 3 different classifiers to get a comprehensive view."""
            }
        )

    def __build_long_term_memory(self):
        pass
