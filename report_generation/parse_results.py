import re

from agents.llm_tools import CVE
from agents.recommendation_agent import SecurityProduct


def parse_classifier_results(results):
    classifiers = []
    majority_vote = {}

    for line in results.split("\n"):
        if "🔹" in line and ":" in line:
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
            majority_vote["agreement_ratio"] = float(line.split(":")[1].strip()) * 100

    return {"classifiers": classifiers, "majority_vote": majority_vote}


def parse_arxiv_results(documents):
    papers = []

    try:
        print(f"\nProcessing {len(documents)} ArXiv documents")

        for doc in documents:
            try:
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


def parse_cve_results(vulnerabilities):
    if vulnerabilities and isinstance(vulnerabilities[0], CVE):
        return vulnerabilities[:5]

    cves = []
    for vuln in vulnerabilities[:5]:
        try:
            cve = vuln["cve"]
            cve_id = cve["id"]

            if not re.match(r"^CVE-\d{4}-\d{4,7}$", cve_id):
                continue

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


def parse_product_results(products):
    if products and isinstance(products[0], SecurityProduct):
        return products[:5]

    return [
        SecurityProduct(
            name=product["name"],
            type=product["type"],
            description=product["description"],
            relevance_score=calculate_product_relevance(product),
        )
        for product in products[:5]
    ]


def calculate_product_relevance(product, security_need):
    relevance = 0
    need_lower = security_need.lower()
    desc_lower = product["description"].lower()

    if any(word in product["name"].lower() for word in need_lower.split()):
        relevance += 3

    if any(word in product["type"].lower() for word in need_lower.split()):
        relevance += 2

    if any(word in desc_lower for word in need_lower.split()):
        relevance += 2

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
