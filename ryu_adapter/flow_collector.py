# /Users/shubh/Documents/Professor Work/cyberwarrior-llm-challenge-main/ryu_adapter/flow_collector.py
import requests
import logging
import pandas as pd
from flow_to_feature_dict import (
    ryu_flow_to_feature_dict,
    ORDERED_FEATURE_NAMES,
)
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_flow_stats(dpid):
    url = f"http://localhost:8080/stats/flow/{dpid}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get(str(dpid), [])
    except Exception as e:
        logger.error(f"Error fetching flow stats: {e}")
        return []


def get_live_feature_vectors_from_ryu(dpid=1) -> pd.DataFrame:
    logger.info(f"Fetching flow stats from Ryu for DPID {dpid}...")
    ryu_flows = get_flow_stats(dpid)
    ryu_flows = sorted(ryu_flows, key=lambda x: x.get("packet_count", 0), reverse=True)

    logger.info(f"Fetched {len(ryu_flows)} flows.")
    feature_vectors = []
    valid_flows = 0


    for ryu_flow in ryu_flows:
        match = ryu_flow.get("match", {})

        # Filter for flows with identifiable IP-level features
        if not any(k in match for k in ["ipv4_src", "nw_src"]):
            print('No match')
            # continue

        try:
            feature_dict = ryu_flow_to_feature_dict(ryu_flow)
            feature_row = [feature_dict[name] for name in ORDERED_FEATURE_NAMES]
            feature_vectors.append(feature_row)
            valid_flows += 1
        except Exception as e:
            logger.error(f"Failed to process flow: {e}")
            continue

    if not feature_vectors:
        logger.warning("No valid flow entries with usable features.")
        return pd.DataFrame(columns=ORDERED_FEATURE_NAMES + ["Label"])

    df = pd.DataFrame(feature_vectors, columns=ORDERED_FEATURE_NAMES)
    df["Label"] = 0  # Placeholder
    logger.info(f"✅ Extracted {valid_flows} valid feature rows.")
    return df


if __name__ == "__main__":
    df = get_live_feature_vectors_from_ryu()
    if not df.empty:
        print("Feature DataFrame from Ryu:\n")
        print(df.to_string())
        print("\nCSV Format:\n")
        print(df.to_csv(index=False))
    else:
        print("No feature DataFrame received or an error occurred.")
