import json
import os
from pathlib import Path
import re
from pydantic import BaseModel, Field


from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from agents.knowledge_source import KnowledgeSource

from agents.llm_tools import (
    safe_web_search_tool,
    safe_arxiv_retrieve_tool,
)


class SecurityProduct(BaseModel):
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


class RecommendationAgent:
    def __init__(self):
        # Long term memory dataset
        self.ltm_db = KnowledgeSource(
            Path("agents/recommendation_agent_long_term_memory"),
        )

        self.__initialize_tools()
        self.__initialize_llm()

    def __initialize_tools(self):
        self.tools = [
            safe_web_search_tool,
            safe_arxiv_retrieve_tool,
        ]

    def __initialize_llm(self):
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

        prompt = PromptTemplate.from_template(
            """You are an expert IoT traffic analyzer and security agent. Your task is 
            to provide product recommendations that can help prevent future attacks and 
            potential threats.

            You have access to the following tools:
            {tools}

            Use these tools to analyze the traffic and provide detailed insights about:
            1. Latest research and knowledge about detected threats
            2. Recommendations for network security
            3. Specific product solutions that could help address identified threats

            Remember to:
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

    def get_llm_product_recommendation(self, search_query: str):
        try:
            print(f"\nSearching for security products related to: {search_query}")

            search_terms = [
                f"{search_query} enterprise security solutions reviews",
                f"top rated {search_query} security products comparison",
                f"best enterprise {search_query} security vendors",
            ]

            all_results = []
            for term in search_terms:
                try:
                    results = safe_web_search_tool.run(term)
                    if results:
                        all_results.append(results)
                        print(f"Found results for: {term}")
                except Exception as e:
                    print(f"Error searching '{term}': {str(e)}")
                    continue

            if not all_results:
                print("No search results found")
                return []

            combined_results = "\n\n".join(all_results)

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

            response = self.agent_executor.invoke(prompt)

            try:
                if isinstance(response, str):
                    cleaned_response = re.sub(
                        r"^```json\s*|\s*```$", "", response.strip()
                    )
                    products_data = json.loads(cleaned_response)
                else:
                    products_data = response

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
