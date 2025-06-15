from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_core.tools import Tool

import requests
import re
from pydantic import BaseModel, Field
from typing import List


class CVE(BaseModel):
    id: str = Field(description="CVE ID in format CVE-YYYY-NNNNN")
    url: str = Field(description="URL to the NVD database entry")
    severity: str = Field(description="Severity level (CRITICAL, HIGH, MEDIUM, LOW)")
    score: str = Field(description="CVSS score")
    description: str = Field(description="Description of the vulnerability")


# Initialize DuckDuckGo search
web_search = DuckDuckGoSearchRun()
# Initialize arXiv retriever
arxiv_retriever = ArxivRetriever(
    load_max_docs=5,
    doc_content_chars_max=None,
    max_retries=3,
)


def safe_web_search(query: str) -> str:
    """
    safe_web_search uses the DuckDuckGo search engine to perform a web search.

    Works best with short queries, ideally under 20 words.

    Args:
        query (str): The web search query.

    Returns:
        str: The result of the web search.
    """
    try:
        return web_search.run(query)
    except Exception as e:
        return f"ERROR: Web Search failed: {e}"


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
