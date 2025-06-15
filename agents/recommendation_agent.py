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
    url: str = Field(
        description="Official website or product page URL",
        default="https://www.google.com/search?q=",
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
                    "url": "https://www.paloaltonetworks.com/",
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

            # Create more generic and effective search terms
            if "attack prevention" in search_query.lower():
                attack_type = search_query.replace(" attack prevention", "").strip()
                search_terms = [
                    f"cybersecurity products {attack_type} protection",
                    f"network security solutions enterprise",
                    f"intrusion detection prevention systems",
                ]
            else:
                search_terms = [
                    f"cybersecurity products {search_query}",
                    f"enterprise security solutions",
                    f"network security vendors comparison",
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
                print("No search results found, using fallback recommendations")
                return self._get_fallback_products(search_query)

            # Parse search results to extract URLs directly
            url_mapping = self._extract_urls_from_search_results(all_results)

            combined_results = "\n\n".join(all_results)

            prompt = f"""Extract the top security products/solutions from this text. Focus on enterprise-grade security solutions.

                    Search Results:
                    {combined_results}

                    Return a JSON array of the top 5 most relevant products. Each product should have:
                    - name: The product name (must be real)
                    - type: Category (Network Security, IDS/IPS, Firewall, etc.)
                    - description: Key features and capabilities
                    - relevance_score: 0-10 based on relevance to {search_query}

                    DO NOT include URLs - they will be added separately.

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
                        product_name = product_data["name"]
                        # Find matching URL from search results
                        product_url = self._find_matching_url(product_name, url_mapping)

                        product = SecurityProduct(
                            name=product_name,
                            type=product_data["type"],
                            description=product_data["description"],
                            url=product_url,
                            relevance_score=float(product_data["relevance_score"]),
                        )
                        products.append(product)
                        print(f"Added product: {product.name} -> {product.url}")
                    except Exception as e:
                        print(f"Error parsing product: {str(e)}")
                        continue

                return sorted(products, key=lambda x: x.relevance_score, reverse=True)

            except Exception as e:
                print(f"Error parsing products: {str(e)}")
                return []

        except Exception as e:
            print(f"Error in product recommendations: {str(e)}")
            return self._get_fallback_products(search_query)

    def _get_fallback_products(self, search_query: str):
        """Provide fallback product recommendations when search fails"""
        try:
            # Generic fallback products for different attack types
            if "ddos" in search_query.lower() or "icmp" in search_query.lower():
                return [
                    SecurityProduct(
                        name="Cloudflare DDoS Protection",
                        type="DDoS Protection",
                        description="Cloud-based DDoS protection service that mitigates large-scale distributed denial-of-service attacks at the network edge.",
                        url="https://www.cloudflare.com/ddos/",
                        relevance_score=9.2,
                    ),
                    SecurityProduct(
                        name="AWS Shield Advanced",
                        type="DDoS Protection",
                        description="Managed DDoS protection service that safeguards applications running on AWS against larger and more sophisticated attacks.",
                        url="https://aws.amazon.com/shield/",
                        relevance_score=8.8,
                    ),
                ]
            elif (
                "dictionary" in search_query.lower()
                or "brute force" in search_query.lower()
            ):
                return [
                    SecurityProduct(
                        name="Fail2ban",
                        type="Intrusion Prevention",
                        description="Log-parsing application that protects Linux and Unix servers from brute-force attacks by monitoring log files and blocking suspicious IP addresses.",
                        url="https://github.com/fail2ban/fail2ban",
                        relevance_score=8.5,
                    ),
                    SecurityProduct(
                        name="CrowdStrike Falcon",
                        type="Endpoint Protection",
                        description="Cloud-native endpoint protection platform with advanced threat detection and response capabilities including brute force attack prevention.",
                        url="https://www.crowdstrike.com/en-us/platform/",
                        relevance_score=9.0,
                    ),
                ]
            else:
                # Generic network security products
                return [
                    SecurityProduct(
                        name="Palo Alto Networks Next-Generation Firewall",
                        type="Network Security",
                        description="Enterprise-grade firewall with advanced threat prevention, intrusion detection, and application-level security features.",
                        url="https://www.paloaltonetworks.com/",
                        relevance_score=9.3,
                    ),
                    SecurityProduct(
                        name="Snort IDS/IPS",
                        type="Intrusion Detection/Prevention",
                        description="Open source network intrusion detection and prevention system capable of performing real-time traffic analysis and packet logging.",
                        url="https://www.snort.org/",
                        relevance_score=8.7,
                    ),
                    SecurityProduct(
                        name="Splunk Enterprise Security",
                        type="SIEM",
                        description="Security information and event management platform that provides real-time monitoring, advanced analytics, and incident response capabilities.",
                        url="https://www.splunk.com/en_us/products/enterprise-security.html",
                        relevance_score=8.9,
                    ),
                ]
        except Exception as e:
            print(f"Error in fallback products: {str(e)}")
            return []

    def _extract_urls_from_search_results(self, search_results_list):
        """Extract URLs and titles from search results"""
        url_mapping = {}

        for search_results in search_results_list:
            # Parse individual results from the formatted search string
            results = search_results.split("\n\n")

            for result in results:
                if result.strip():
                    lines = result.strip().split("\n")
                    title = ""
                    url = ""

                    for line in lines:
                        if line.startswith("Title: "):
                            title = line.replace("Title: ", "").strip()
                        elif line.startswith("URL: "):
                            url = line.replace("URL: ", "").strip()

                    if title and url:
                        url_mapping[title.lower()] = url

        print(f"Extracted {len(url_mapping)} URLs from search results")
        return url_mapping

    def _find_matching_url(self, product_name, url_mapping):
        """Find the best matching URL for a product name"""
        product_name_lower = product_name.lower()

        # Try exact match first
        if product_name_lower in url_mapping:
            return url_mapping[product_name_lower]

        # Try partial matches
        for title, url in url_mapping.items():
            # Check if product name is contained in the title
            if any(
                word in title for word in product_name_lower.split() if len(word) > 3
            ):
                return url

            # Check if title contains the product name
            if product_name_lower.replace(" ", "") in title.replace(" ", ""):
                return url

        # Fallback to Google search
        return f"https://www.google.com/search?q={product_name.replace(' ', '+')}"
