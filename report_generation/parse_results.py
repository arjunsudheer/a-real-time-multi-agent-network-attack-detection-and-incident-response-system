import re

from agents.llm_tools import CVE
from agents.recommendation_agent import SecurityProduct


def parse_classifier_results(results):
    # Handle detailed results structure from network_agent_system.py
    if isinstance(results, dict) and "pre_detection" in results:
        classifiers = []
        
        # Add pre-detection classifier
        pre_det = results["pre_detection"]
        classifiers.append({
            "name": "Pre-Detection",
            "prediction": pre_det["prediction"],
            "confidence": pre_det["confidence"]
        })
        
        # Add post-classification classifiers if available
        if "post_classification" in results and results["post_classification"]["classifiers"]:
            for clf in results["post_classification"]["classifiers"]:
                classifiers.append({
                    "name": clf["name"],
                    "prediction": clf["prediction"],
                    "confidence": clf["confidence"]
                })
            
            # Use real majority vote results
            majority_vote = results["post_classification"]["majority_vote"]
        else:
            # If no post-classification, use pre-detection as majority vote
            majority_vote = {
                "prediction": pre_det["prediction"],
                "agreement_ratio": 100.0
            }
        
        # Add LLM classifier if it was used
        if results.get("used_llm", False):
            classifiers.append({
                "name": "LLM Agent",
                "prediction": results["final_prediction"],
                "confidence": 88.0
            })
        
        return {"classifiers": classifiers, "majority_vote": majority_vote}
    
    # Fallback for simple string format
    prediction = results.strip() if isinstance(results, str) else str(results).strip()
    
    classifiers = [{
        "name": "Network Agent System",
        "prediction": prediction,
        "confidence": 95.0
    }]
    
    majority_vote = {
        "prediction": prediction,
        "agreement_ratio": 100.0
    }

    return {"classifiers": classifiers, "majority_vote": majority_vote}


def parse_arxiv_results(documents):
    papers = []

    try:
        # Handle case where documents might be a string (from tool) instead of document objects
        if isinstance(documents, str):
            print(f"Received string instead of document objects: {documents[:200]}...")
            return []

        if not documents:
            print("No documents to process")
            return []

        print(f"\nProcessing {len(documents)} ArXiv documents")

        for doc in documents:
            try:
                # Check if doc has metadata attribute
                if not hasattr(doc, 'metadata'):
                    print(f"- Document missing metadata attribute: {type(doc)}")
                    continue

                metadata = doc.metadata
                print(f"Available fields in metadata: {list(metadata.keys())}")

                # Safely extract fields with fallbacks
                title = metadata.get("Title", "Unknown Title")
                authors = metadata.get("Authors", "Unknown Authors")
                entry_id = metadata.get("Entry ID", "")
                published = metadata.get("Published", None)
                
                # Handle arxiv_id extraction safely
                arxiv_id = entry_id.split("/")[-1] if entry_id else "unknown"
                
                # Handle published date safely
                published_str = published.isoformat() if published and hasattr(published, 'isoformat') else str(published) if published else "Unknown"

                paper = {
                    "title": title,
                    "authors": authors,
                    "arxiv_id": arxiv_id,
                    "url": entry_id,
                    "published": published_str,
                    "summary": getattr(doc, 'page_content', 'No summary available'),
                }

                papers.append(paper)
                print(f"- Successfully parsed paper: {paper['title'][:100]}")

            except Exception as e:
                print(f"- Error parsing paper: {str(e)}")
                continue

        return papers

    except Exception as e:
        print(f"Error in parse_arxiv_results: {str(e)}")
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


def parse_product_results(products, security_need="general security"):
    if products and isinstance(products[0], SecurityProduct):
        return products[:5]

    return [
        SecurityProduct(
            name=product["name"],
            type=product["type"],
            description=product["description"],
            relevance_score=calculate_product_relevance(product, security_need),
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
