import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import re

# For faithfulness scoring with DeBERTa NLI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# For relevance scoring with nomic-embed-text
try:
    import ollama
    ollama_available = True
except ImportError:
    print("Ollama is not installed. Relevance scoring with nomic-embed-text will be skipped.")
    ollama = None
    ollama_available = False

# Import IDSAgent from the agents directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.network_agent_demo import IDSAgent

RESULTS_PATH = Path(__file__).parent / "results/llm_metrics_results.csv"
SAMPLE_FRACTION = 0.01  # 1% of the test data

# Load DeBERTa NLI model and tokenizer
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def clean_for_relevance(text):
    if not text:
        return ""
    # Remove non-ASCII characters (emojis, etc.)
    text = text.encode("ascii", "ignore").decode()
    # Remove non-alphanumeric except basic punctuation
    text = re.sub(r'[^\w\s.,:;!?-]', '', text)
    # Try to extract the main prediction after 'Final Prediction:'
    match = re.search(r'Final Prediction:\s*([^\n]+)', text)
    if match:
        return match.group(1).strip()
    # Fallback: return the first sentence or first 200 chars
    return text.strip().split('\n')[0][:200]

def compute_nli_faithfulness(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # Label mapping: 0=entailment, 1=neutral, 2=contradiction
    probs = torch.softmax(logits, dim=1)[0]
    entailment_prob = float(probs[0])  # Faithfulness = entailment probability
    return entailment_prob

def compute_relevance(query, output):
    if not ollama_available or not query or not output:
        return None
    try:
        emb_query = np.array(ollama.embeddings(model='nomic-embed-text', prompt=str(query))['embedding'])
        emb_output = np.array(ollama.embeddings(model='nomic-embed-text', prompt=str(output))['embedding'])
        score = float(np.dot(emb_query, emb_output) / (np.linalg.norm(emb_query) * np.linalg.norm(emb_output)))
        return score
    except Exception as e:
        print(f"Ollama relevance error: {e}")
        return None

def main():
    # Load a random sample (1%) from test.csv
    print("Loading test data sample...")
    test_df = pd.read_csv("test.csv")
    sample_size = min(20, len(test_df))
    samples = test_df.sample(n=sample_size, random_state=42)

    results = []
    for idx, (i, row) in enumerate(samples.iterrows()):
        true_label = row["Label"]
        query_str = str(true_label)
        agent = IDSAgent(return_context=True, query=query_str)
        try:
            llm_output, retrieved_context = agent.analyze(row.to_frame().T)
        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
            llm_output = "ERROR"
            retrieved_context = ""
        parsed = agent._parse_classifier_results(agent.analyze_with_majority_vote())
        majority_pred = parsed["majority_vote"].get("prediction", "")
        agreement_ratio = parsed["majority_vote"].get("agreement_ratio", None)
        classifier_preds = parsed["classifiers"]
        classifier_metrics = {}
        for clf in classifier_preds:
            name = clf["name"]
            classifier_metrics[f"{name}_prediction"] = clf["prediction"]
            classifier_metrics[f"{name}_confidence"] = clf["confidence"]
        faithfulness = compute_nli_faithfulness(retrieved_context, llm_output)
        query_context = agent.build_context(row.to_frame().T)
        cleaned_query = clean_for_relevance(query_context)
        cleaned_output = clean_for_relevance(llm_output)
        relevance = compute_relevance(cleaned_query, cleaned_output)
        result = {
            "sample_index": i,
            "true_label": true_label,
            "majority_prediction": majority_pred,
            "agreement_ratio": agreement_ratio,
            "llm_output": llm_output,
            "retrieved_context": retrieved_context,
            "relevance": relevance,
            "faithfulness": faithfulness,
            **classifier_metrics,
        }
        results.append(result)
    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved LLM metrics results to {RESULTS_PATH}")

if __name__ == "__main__":
    main() 