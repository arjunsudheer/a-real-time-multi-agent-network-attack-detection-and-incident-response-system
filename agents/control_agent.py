import os
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from agents.knowledge_source import KnowledgeSource

from agents.llm_tools import (
    safe_web_search_tool,
    safe_arxiv_retrieve_tool,
    safe_cve_search_tool,
)


class ControlAgent:
    def __init__(self):
        # Long term memory dataset
        self.ltm_db = KnowledgeSource(
            Path("agents/control_agent_long_term_memory"),
        )

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_tools(self):
        self.tools = [
            safe_web_search_tool,
            safe_arxiv_retrieve_tool,
            safe_cve_search_tool,
        ]

    def __initialize_llm(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

        prompt = PromptTemplate.from_template(
            """You are an expert IoT traffic analyzer and security agent. Your task is to analyze network traffic patterns 
            from multiple machine learning classifiers and provide insights about potential threats.

            You have access to the following tools:
            {tools}

            Use these tools to analyze the traffic and provide detailed insights about:
            1. Potential threats and their severity
            2. Known CVE vulnerabilities related to detected threats
            3. Latest research and knowledge about detected threats
            4. Recommendations for network security
            5. Specific product solutions that could help address identified threats

            Remember to:
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

        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=300,
            early_stopping_method="generate",
        )
