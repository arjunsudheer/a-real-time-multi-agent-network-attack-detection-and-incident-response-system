from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

import numpy as np
import pandas as pd
import os


class FeatureSelectionAgent:
    def __init__(self, df: pd.DataFrame, label_column: str, dataset_name: str) -> None:
        self.df = df
        self.label_column = label_column
        self.dataset_name = dataset_name

        # Ignore the label column and index column for feature selection
        self.dataset_sample = df.drop([label_column], axis=1)
        self.dataset_sample.reset_index(drop=True, inplace=True)
        self.dataset_features = list(self.dataset_sample.columns)

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_llm(self) -> None:
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

        prompt = PromptTemplate.from_template(
            """You are an expert IoT traffic analyzer and security agent. Your task is to decide which features
            are most important to keep in a dataset, and which features can be dropped. Your goal is to select the
            most relevant features in the dataset such that there is a meaningful reduction in training and inference
            time with no drop in performance.

            You have access to the following tools:
            {tools}

            Remember to:
            - Understand the dataset contents
            - Look for features that may be redundant or irrelevant
            - Search for latest information about detected threats
            - Reference relevant academic research when available
            - Provide clear explanations of your reasoning

            To use a tool, please use the following format:
            Thought: I need to analyze this dataset
            Action: the action to take, should be one of [{tool_names}]
            Action Input: (leave empty for viewing the dataset, provide search terms for WebSearch and AcademicLiteratureSearch)
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
            max_iterations=10,
            max_execution_time=300,  # 5 minutes timeout
        )

    def __initialize_tools(self) -> None:
        # Initialize DuckDuckGo search
        web_search = DuckDuckGoSearchRun()
        # Initialize arXiv retriever
        arxiv_retriever = ArxivRetriever()

        def view_first_hundred_rows() -> pd.Series:
            return self.dataset_sample.head(n=100)

        def view_unique_labels() -> np.ndarray:
            return self.df[self.label_column].unique()

        def view_features() -> list:
            return self.dataset_features

        def drop_irrelevant_features(feature_rankings: str) -> str:
            invalid_features = []

            rankings = feature_rankings.split(",")

            for ranking in rankings:
                feature, relevance_score = ranking.split(":")

                if feature not in self.dataset_features:
                    invalid_features.append(feature)
                elif float(relevance_score) < 0.3:
                    # Drop features
                    self.df.drop(columns=[feature], inplace=True)

            return (
                f"Finished dropping invalid features!"
                if len(invalid_features) == 0
                else f"The following features do not exist: {invalid_features}. Perhaps you made a spelling mistake? Call the tool again with the corrected feature names."
            )

        def safe_web_search(query: str) -> str:
            print("web search called")
            try:
                return web_search.run(query)
            except Exception as e:
                return f"ERROR: Web Search failed: {e}"

        def safe_knowledge_retrieve(query: str) -> str:
            print("arxiv called")
            try:
                return arxiv_retriever.run(query)
            except Exception as e:
                return f"ERROR: Academic Literature Search failed: {e}"

        # Create unified classifier tool
        self.tools = [
            Tool(
                name="ViewDataset",
                func=lambda _: view_first_hundred_rows(),
                description="Shows the first 100 rows of the dataset. You do not need to pass any inputs, just call the tool.",
            ),
            Tool(
                name="ViewUniqueLabels",
                func=lambda _: view_unique_labels(),
                description="Shows the unique elements in the dataset's label column. You do not need to pass any inputs, just call the tool.",
            ),
            Tool(
                name="ViewFeatures",
                func=lambda _: view_features(),
                description="Shows the features in the dataset. You do not need to pass any inputs, just call the tool.",
            ),
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
            Tool(
                name="RemoveUnnecessaryFeatures",
                func=lambda features_to_drop: drop_irrelevant_features(
                    features_to_drop
                ),
                description="Use this as the **final step**. You need to pass an argument in the following format"
                "feature_name:rating"
                "feature_name is the feature name, and rating is the feature importance score on a scale of 0 (irrelevant) to 1 (relevant)."
                "Make sure that you pass each feature and rating as a comma-separated string.",
            ),
        ]

    def select_features(self) -> list[str]:
        # Prompt the LLM agent for each feature to see if is it worth keeping or dropping
        self.agent_executor.invoke(
            {
                "input": f"""
                You must execute these steps in this exact order:
                1. Look at the dataset to understand how each feature is represented (e.g. categorical vs. numerical)
                2. Look at all the unique labels that exist in the dataset to understand what types of network attacks are being classified
                3. Look at all the features in the dataset to understand what features are available.
                4. Refer to existing academic literature on the importance of specific features in the specified dataset.
                5. Search the web to learn more about the relevance of some features on that dataset.
                6. Provide a feature relevance score from 0 (irrelevant) to 1 (relevant) for each feature in the dataset. Make sure you call the
                tool to drop the irrelevant features at the very end.

                Now analyze the {self.dataset_name} dataset to find our which features are important, and which features are irrelevant.
                """
            }
        )

        return self.df
