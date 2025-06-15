from langchain_community.retrievers import ArxivRetriever
from langchain_core.tools import Tool

import requests
import re
import os
import time
from pydantic import BaseModel, Field
from typing import List

# Rate limiting for Brave Search API
_last_search_time = 0


class CVE(BaseModel):
    id: str = Field(description="CVE ID in format CVE-YYYY-NNNNN")
    url: str = Field(description="URL to the NVD database entry")
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW)")
    score: str = Field(description="CVSS score")
    description: str = Field(description="Description of the vulnerability")


# Initialize arXiv retriever
arxiv_retriever = ArxivRetriever(
    load_max_docs=5,
    doc_content_chars_max=None,
    max_retries=3,
)


def safe_web_search(query: str) -> str:
    """
    safe_web_search uses the Brave Search API to perform a web search.

    Uses direct HTTP requests to Brave Search API.
    Works best with short queries, ideally under 20 words.
    Free tier provides 2000 searches per month.
    Rate limited to 1 request per second.
    Get your API key from: https://api.search.brave.com/

    Args:
        query (str): The web search query.

    Returns:
        str: The result of the web search.
    """
    global _last_search_time

    try:
        # Rate limiting: ensure at least 1 second between requests
        current_time = time.time()
        time_since_last_request = current_time - _last_search_time
        if time_since_last_request < 1.0:
            sleep_time = 1.0 - time_since_last_request
            time.sleep(sleep_time)

        _last_search_time = time.time()
        # Get API key from environment or use fallback
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            # Fallback API key for demo purposes - replace with your own
            api_key = "BSAqaEHMsVKdqArHgUr_1Po3vZNVj8T"

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        params = {
            "q": query,
            "count": 10,
            "text_decorations": False,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
        }

        print(f"Brave Search - Making request to: {url}")
        print(f"Brave Search - Query: {query}")
        print(f"Brave Search - Using API key: {api_key[:10]}...")

        response = requests.get(url, headers=headers, params=params, timeout=10)

        print(f"Brave Search - Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Brave Search - Response keys: {list(data.keys())}")

            results = []

            if "web" in data and "results" in data["web"]:
                web_results = data["web"]["results"]
                print(f"Brave Search - Found {len(web_results)} web results")

                for result in web_results:
                    title = result.get("title", "")
                    url = result.get("url", "")
                    description = result.get("description", "")

                    formatted_result = (
                        f"Title: {title}\nURL: {url}\nDescription: {description}\n"
                    )
                    results.append(formatted_result)
            else:
                print(f"Brave Search - No 'web' results found in response")
                print(f"Brave Search - Full response: {data}")

            if results:
                print(f"Brave Search - Returning {len(results)} formatted results")
                return "\n".join(results)
            else:
                print("Brave Search - No search results found")
                return "No search results found."

        elif response.status_code == 429:
            print(f"Brave Search - Rate limit exceeded")
            return (
                "ERROR: Brave Search API rate limit exceeded. Please try again later."
            )
        elif response.status_code == 401:
            print(f"Brave Search - Invalid API key")
            try:
                error_data = response.json()
                print(f"Brave Search - Error response: {error_data}")
            except:
                print(f"Brave Search - Error response text: {response.text}")
            return "ERROR: Invalid Brave Search API key. Please set BRAVE_API_KEY environment variable or get a key from https://api.search.brave.com/"
        else:
            print(f"Brave Search - Unexpected status code: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Brave Search - Error response: {error_data}")
            except:
                print(f"Brave Search - Error response text: {response.text}")
            return (
                f"ERROR: Brave Search API returned status code {response.status_code}"
            )

    except Exception as e:
        return f"ERROR: Brave Search failed: {str(e)}"


def safe_arxiv_retrieve(query: str) -> str:
    """
    safe_arxiv_retrieve uses the arxiv retrieval tool to search academic literature.

    Args:
        query (str): The academic literature search query.

    Returns:
        str: The result of the academic literature search.
    """
    try:
        return arxiv_retriever.run(query)
    except Exception as e:
        return f"ERROR: Academic Literature Search failed: {e}"


safe_web_search_tool = Tool(
    name="WebSearch",
    func=lambda query: safe_web_search(query),
    description=safe_web_search.__doc__,
)
safe_arxiv_retrieve_tool = Tool(
    name="AcademicLiteratureSearch",
    func=lambda query: safe_arxiv_retrieve(query),
    description=safe_arxiv_retrieve.__doc__,
)


def safe_search_cve(query: str) -> List[CVE]:
    def get_alternative_vulnerability_data(query: str) -> List[CVE]:
        try:
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
                    results = safe_web_search_tool.run(
                        f"{search_term} site:cve.mitre.org OR site:nvd.nist.gov"
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"Error searching term '{search_term}': {str(e)}")

            combined_results = "\n".join(all_results)

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

                start = max(0, match.start() - 200)
                end = min(len(combined_results), match.end() + 200)
                context = combined_results[start:end]

                severity = "MEDIUM"
                context_lower = context.lower()
                for sev, indicators in severity_indicators.items():
                    if any(ind in context_lower for ind in indicators):
                        severity = sev
                        break

                sentences = re.split(r"[.!?]+", context)
                description = ""
                for sentence in sentences:
                    if cve_id in sentence:
                        description = sentence.strip()
                        next_idx = sentences.index(sentence) + 1
                        if next_idx < len(sentences):
                            description += ". " + sentences[next_idx].strip()
                        break

                if not description:
                    description = f"Vulnerability related to {query} attacks"

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

                if len(cves) >= 5:
                    break

            return cves

        except Exception as e:
            print(f"Error in alternative vulnerability search: {str(e)}")
            return []

    try:
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

        # api_key = os.getenv("NVD_API_KEY")
        api_key = "ea20866b-6e87-46aa-a1d2-1d4a7bdede0"

        clean_query = re.sub(r"[^\w\s-]", "", query)
        if clean_query.lower() == "icmp flood":
            search_terms = ["ICMP flood", "ping flood", "ICMP DoS"]
        elif clean_query.lower() == "benign":
            return []
        else:
            search_terms = [clean_query, "network attack", "network security"]

        keyword_search = " OR ".join(f'"{term}"' for term in search_terms)

        params = {
            "keywordSearch": keyword_search,
            "resultsPerPage": 10,
            "startIndex": 0,
        }

        headers = {
            "User-Agent": "IDS-Analysis-Tool/1.0",
            "Content-Type": "application/json",
        }
        if api_key:
            headers["apiKey"] = api_key

        print(f"\nSearching CVEs with query: {keyword_search}")

        response = requests.get(base_url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()

            total_results = data.get("totalResults", 0)
            if total_results == 0:
                print(f"No CVEs found for query: {keyword_search}")
                return get_alternative_vulnerability_data(query)

            vulnerabilities = data.get("vulnerabilities", [])
            cves = []

            for vuln in vulnerabilities[:5]:
                cve_data = vuln.get("cve", {})

                cve_id = cve_data.get("id")
                if not cve_id or not re.match(r"^CVE-\d{4}-\d{4,7}$", cve_id):
                    continue

                metrics = cve_data.get("metrics", {})
                cvss_data = None
                severity = "UNKNOWN"
                score = "N/A"

                if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
                    cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
                    severity = cvss_data.get("baseSeverity", "UNKNOWN")
                    score = str(cvss_data.get("baseScore", "N/A"))
                elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
                    cvss_data = metrics["cvssMetricV30"][0].get("cvssData", {})
                    severity = cvss_data.get("baseSeverity", "UNKNOWN")
                    score = str(cvss_data.get("baseScore", "N/A"))
                elif "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
                    cvss_data = metrics["cvssMetricV2"][0].get("cvssData", {})
                    base_score = cvss_data.get("baseScore", 0)
                    score = str(base_score)
                    if base_score >= 7.0:
                        severity = "HIGH"
                    elif base_score >= 4.0:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

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
            if response.status_code == 403:
                print("Access denied. API key may be required or invalid.")
            elif response.status_code == 429:
                print(
                    "Rate limit exceeded. Consider using an API key or waiting before retrying."
                )
            return get_alternative_vulnerability_data(query)

    except Exception as e:
        print(f"Error accessing NVD API: {str(e)}")
        return get_alternative_vulnerability_data(query)


safe_cve_search_tool = Tool(
    name="CVESearch",
    func=safe_search_cve,
    description="Search for CVE vulnerabilities related to IoT devices or protocols. Input should be a search term like 'IoT camera vulnerability' or 'MQTT protocol'",
)
