from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from pathlib import Path
import os
import time
import numpy as np
import pandas as pd

from agents.knowledge_source import KnowledgeSource
from agents.llm_tools import safe_web_search_tool, safe_arxiv_retrieve_tool


class LabelingAgent:
    def __init__(self, dataset_directory: Path, label_column: str) -> None:
        """
        __init__ initializes the dataset directory, label column, and labeling agent.

        Initializes the long-term memory KnowledgeSource for RAG. Initializes the labeling agent and
        tools for the labeling agent to call.

        Args:
            dataset_directory (Path): The path to the parent directory containing the dataset.
            label_column (str): The name of the column in the dataset containing the labels.
        """
        self.dataset_directory = dataset_directory
        self.label_column = label_column

        # Long term memory dataset
        self.ltm_db = KnowledgeSource(
            Path("agents/labeling_agent_long_term_memory"),
        )

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_llm(self) -> None:
        """
        __initialize_llm initializes the labeling agent using the Gemini 2.0 Flash model.

        Creates a reAct agent with access to tools.
        """
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
        prompt = PromptTemplate.from_template(
            """You are an expert IoT traffic analyzer and security agent. Your task is to analyze network traffic patterns 
            and provide insights about potential threats.

            You have access to the following tools:
            {tools}

            Use these tools to analyze the traffic and provide detailed insights about:
            1. Potential threats and their severity
            2. Latest research and knowledge about detected threats
            3. General web search to learn more about how particular attributes correspond to an attack type

            Remember to:
            - Look for patterns in the predictions
            - Search for latest information about detected threats
            - Reference relevant academic research when available
            - Provide clear explanations of your reasoning

            To use a tool, please use the following format:
            Thought: I need to analyze this sample with multiple classifiers
            Action: the action to take, should be one of [{tool_names}]
            Action Input: (leave empty for ViewUniqueLabels, provide search terms for WebSearch and AcademicLiteratureSearch)
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
        """
        __initialize_tools initializes the tools for the labeling agent to use.

        Creates a unified tool for the labeling agent.

        Args:
            include_ltm (bool, optional): Whether or not the long-term memory RAG should be
            provided as a tool. Should be set to False during long-term memory cache building,
            and True during inference. Defaults to False.
        """

        def view_unique_labels() -> np.ndarray:
            """
            view_unique_labels retrieves the unique labels from the ACI-IoT-2023 dataset.

            Returns:
                np.ndarray: The unique labels found in the ACI-IoT-2023 dataset.
            """
            df = pd.read_csv(
                self.dataset_directory / "original_dataset" / "ACI-IoT-2023.csv"
            )
            return df[self.label_column].unique()

        # Create unified classifier tool
        self.tools = [
            Tool(
                name="ViewUniqueLabels",
                func=lambda _: view_unique_labels(),
                description="Shows the unique elements in the dataset's label column. You do not need to pass any inputs, just call the tool.",
            ),
            safe_web_search_tool,
            safe_arxiv_retrieve_tool,
        ]

        # Only allow the long-term memory access tool to be used during inference
        # This tool will not be available during training since the long-term memory cache has not been created yet
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
        self, sample: np.ndarray, incorrect_labels: set = None
    ) -> dict[str]:
        """
        get_llm_prediction prompts the labeling agent for network attack class prediction.

        Returns:
            dict[str]: The initial prompt, the response of the labeling agent including
            any research it has done, and the final network attack class prediction.
        """
        prompt = f"""Analyze this traffic sample to determine what kind of attack it might be. 
                For each sample:
                1. Search for information about detected threats
                2. Find relevant academic research
                3. As your final output, only mention the attack class, and nothing else
                
                Please make sure to provide a valid attack class (label) for the following sample: {sample}.
                {f"The following labels are incorrect, and should not be predicted: {incorrect_labels}. Do not repeat the same incorrect prediction. Consider alternative attack classes from the known list." if incorrect_labels else ""}"""

        # Run agent
        response = self.agent_executor.invoke({"input": prompt})

        # return response
        return {
            "prompt": prompt,
            "response": response,
            "output": response.get(
                "output", ""
            ).strip(),  # Get the labeling agent predicted label
        }

    def build_long_term_memory(
        self, samples: pd.DataFrame, labels: np.ndarray, rate_limit_threshold: int = 3
    ) -> None:
        """
        build_long_term_memory prompts the labeling agent and saves correct responses into the
        long-term memory KnowledgeSource.

        Saves at-most 25 correct predictions to balance time with an adequate number of entries
        in long-term memory. Keeps re-prompting the labeling agent until it gets the right attack
        class prediction to ensure that there are entries in the long-term memory KnowledgeSource.

        Args:
            samples (pd.DataFrame): The samples to make the attack class predictions on.
            labels (np.ndarray): The correct labels for the samples. Used to verify the
            correctness of the labeling agent's prediction.
            rate_limit_threshold (int, optional): The number of continuous prompts to
            make to the labeling agent before a short timeout. Used to prevent rate
            limit issues. Defaults to 3.
        """
        # Store at most 25 sample predictions so the long-term memory cache building process does not take too long
        for i in range(min(25, len(samples))):
            # Store incorrect predictions to help the LLM make the right prediction
            incorrect_predictions = set()
            # Store the number of attempts to use for preventing rate limits
            attempts = 0

            # Initialize the LLM for every sample to avoid hitting rate limits
            self.__initialize_llm()

            # Get the current sample to be processed
            sample = samples.iloc[i]
            # Get the current label to ensure that the LLM makes the right prediction
            correct_label = labels[i]

            # Get the initial LLM prediction
            llm_response = self.get_llm_prediction(
                sample, incorrect_labels=incorrect_predictions
            )
            predicted_label = llm_response["output"]
            print(f"{predicted_label} vs {correct_label}")

            # Keep prompting the LLM until it makes the correct prediction
            # Ignore the current sample if a duplicate incorrect prediction is made
            while (
                predicted_label != correct_label
                and predicted_label not in incorrect_predictions
            ):
                # Update the incorrect predictions made by the labeling agent
                incorrect_predictions.add(predicted_label)
                attempts += 1

                # Wait for 30 seconds to avoid running into rate limit issues
                if attempts % rate_limit_threshold == 0:
                    time.sleep(30)

                # Re-prompt the LLM until it gets the correct label
                llm_response = self.get_llm_prediction(
                    sample, incorrect_labels=incorrect_predictions
                )
                predicted_label = llm_response["output"]
                print(f"{predicted_label} vs {correct_label}")

            # Update long term memory only when a correct prediction is made
            if predicted_label == correct_label:
                self.ltm_db.add_knowledge(
                    f"""
                    Prompt: {llm_response['prompt']}

                    Final Answer: {llm_response['output']}

                    Trace: {llm_response['response']}
                    """
                )
                print("Added correct prediction to KnowledgeSource")
            else:
                print("Skipped sample after duplicate incorrect prediction")
