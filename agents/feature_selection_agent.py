from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

import numpy as np
import pandas as pd
import os

from agents.llm_tools import safe_web_search_tool, safe_arxiv_retrieve_tool


class FeatureSelectionAgent:
    def __init__(self, df: pd.DataFrame, label_column: str) -> None:
        """
        __init__ initializes the tools for the LLM and the feature selection agent.

        Initializes the DataFrame, label column, dataset name, and prepares the DataFrame
        for further analysis. Initializes the tools for the LLM and the feature selection
        agent itself.

        Args:
            df (pd.DataFrame): The DataFrame to perform feature selection on.
            label_column (str): The name of the label column.
        """
        self.df = df
        self.label_column = label_column

        # Ignore the label column and index column for feature selection
        self.dataset_sample = df.drop([label_column], axis=1)
        self.dataset_sample.reset_index(drop=True, inplace=True)
        self.dataset_features = list(self.dataset_sample.columns)

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_llm(self) -> None:
        """
        __initialize_llm initializes the feature selection agent using the Gemini 2.0 Flash model.

        Creates a reAct agent with access to tools.
        """
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv(
                "GOOGLE_API_KEY", "AIzaSyC72eGdAEHU9ZBAhXJWAg6b8fCQSRmgDBU"
            ),
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
            Action Input: the input to the action. If there is no required input, leave this empty
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
            max_execution_time=300,
        )

    def __initialize_tools(self) -> None:
        """
        __initialize_tools creates a unified classifier tool for the feature selection agent to use.
        """

        def view_unique_labels() -> np.ndarray:
            """
            view_unique_labels returns the unique labels in the dataset.

            Returns:
                np.ndarray: The unique labels in the dataset.
            """
            return self.df[self.label_column].unique()

        def view_features() -> list:
            """
            view_features returns the unique columns (features) in the dataset, excluding the label column.

            Returns:
                list: The unique columns in the dataset, excluding the label column.
            """
            return self.dataset_features

        def drop_irrelevant_features(features_to_drop: str) -> str:
            """
            drop_irrelevant_features drops the provided features if the exist in the DataFrame.

            Splits the string using a comma as the delimiter.

            Args:
                features_to_drop (str): A comma-separated string with the names of the features to drop.

            Returns:
                str: A success message if the features were dropped successfully, or an error
                message with the features that were not able to be dropped.
            """
            invalid_features = []
            valid_features = []

            split_features = features_to_drop.strip().split(",")

            for feature in split_features:
                cleaned_feature = feature.strip()
                if cleaned_feature not in self.dataset_features:
                    invalid_features.append(cleaned_feature)
                else:
                    valid_features.append(cleaned_feature)

            if len(invalid_features) == 0:
                # Drop features
                self.df.drop(columns=valid_features, inplace=True)
                return "Finished dropping invalid features!"
            else:
                f"The following features do not exist: {invalid_features}. Perhaps you made a spelling mistake? Call the tool again with the corrected feature names."

        # Create unified classifier tool
        self.tools = [
            Tool(
                name="ViewUniqueLabels",
                func=lambda _: view_unique_labels(),
                description=view_unique_labels.__doc__,
            ),
            Tool(
                name="ViewFeatures",
                func=lambda _: view_features(),
                description=view_features.__doc__,
            ),
            Tool(
                name="RemoveUnnecessaryFeatures",
                func=lambda features_to_drop: drop_irrelevant_features(
                    features_to_drop
                ),
                description=drop_irrelevant_features.__doc__,
            ),
            safe_web_search_tool,
            safe_arxiv_retrieve_tool,
        ]

    def select_features(self) -> pd.DataFrame:
        """
        select_features prompts the feature selection agent to rank the relevance of each feature and drop the irrelevant features.

        Prompts the feature selection agent to first understand the dataset by viewing the unique labels and features in the dataset.
        Then, the feature selection agent is prompted to do more research on particular features if it needs to learn more about the
        relevance of a given feature in the dataset. Finally, it is asked to rank the relevance of each feature, and drop the features
        that it thinks are irrelevant.

        Returns:
            pd.DataFrame: The DataFrame after the feature selection agent has dropped the irrelevant features.
        """
        # Prompt the LLM agent for each feature to see if is it worth keeping or dropping
        self.agent_executor.invoke(
            {
                "input": f"""
                Given the unique labels that exist in the dataset, drop the features that are irrelevant
                on their own, or those that may cause overfitting. Make sure to consider features that may 
                change often and therefore will not generalize well. If a features seems very high-level but 
                could help distinguish between two different labels that are present in the dataset, then it 
                is most likely useful and should be kept. If certain features are engineered or derived from 
                other features that you want to drop, analyze if the derived feature may help distinguish 
                between different attack classes instead of blindly dropping the derived feature too.

                You must execute these steps in this exact order:

                First, understand the dataset to provide an accurate and tailored response:
                1. Look at all the unique labels that exist in the dataset to understand what types of network attacks are being classified
                2. Look at all the features in the dataset to understand what features are available.

                If any features seem confusing or unclear in relevance, you can:
                3. Refer to existing academic literature to learn about the importance of specific features.
                4. Search the web to learn more about the relevance of some specific features.
                
                Finally, you must always perform this final step:
                5. Provide a feature relevance score from 0 (irrelevant) to 1 (relevant) for each feature in the dataset. Make sure you call the
                tool to drop the irrelevant features at the very end.

                Now analyze the dataset to find our which features are important, and which features are irrelevant.
                """
            }
        )

        return self.df
