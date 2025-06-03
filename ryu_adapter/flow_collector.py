# /Users/shubh/Documents/Professor Work/cyberwarrior-llm-challenge-main/ryu_adapter/flow_collector.py
import requests
import logging
import json  # For pretty printing in main, and explicit exception handling
import pandas as pd

# Attempt to import from ryu_adapter, ensure it's in PYTHONPATH or project structure
from .flow_to_feature_dict import ryu_flow_to_feature_dict, ORDERED_FEATURE_NAMES

# Configure logging for this module
logger = logging.getLogger(__name__)  # Use the module's logger


def get_flow_stats(dpid=1):
    """
    Fetches flow statistics from the Ryu controller's REST API.
    """
    url = f"http://localhost:8080/stats/flow/{dpid}"
    try:
        response = requests.get(url, timeout=5)  # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        # Ryu returns a dictionary where keys are DPIDs as strings
        return data.get(str(dpid), [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching flow stats from Ryu: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from Ryu: {e}")
        return []


def get_live_feature_vectors_from_ryu(num_samples_to_fetch=5, dpid=1) -> pd.DataFrame:
    """
    Fetches live flow data from Ryu, converts to feature vectors,
    and returns a DataFrame ready for analysis (features are raw, unscaled).
    The DataFrame includes a dummy 'Label' column.
    """
    logger.info(f"Fetching live flow stats from Ryu for DPID {dpid}...")
    ryu_flows = get_flow_stats(dpid=dpid)

    if not ryu_flows:
        logger.warning("No flow entries received from Ryu.")
        # Return an empty DataFrame with expected columns if no flows
        return pd.DataFrame(columns=ORDERED_FEATURE_NAMES)

    logger.info(f"Received {len(ryu_flows)} flow entries from Ryu.")

    feature_vectors_list = []
    # Process up to num_samples_to_fetch or all available if fewer
    for ryu_flow_entry in ryu_flows[:num_samples_to_fetch]:
        # ryu_flow_to_feature_dict returns a dictionary of raw feature values
        try:
            feature_dict = ryu_flow_to_feature_dict(ryu_flow_entry)
            # Ensure features are in the ORDERED_FEATURE_NAMES sequence for DataFrame creation
            ordered_feature_values = [
                feature_dict[name] for name in ORDERED_FEATURE_NAMES
            ]
            feature_vectors_list.append(ordered_feature_values)
        except Exception as e:
            logger.error(
                f"Error converting Ryu flow to feature dict: {e}. Flow: {ryu_flow_entry}"
            )
            continue  # Skip this flow

    if not feature_vectors_list:
        logger.warning("Could not convert any Ryu flows to feature vectors.")
        return pd.DataFrame(columns=ORDERED_FEATURE_NAMES)

    # Create DataFrame with the correct feature names (these are raw features)
    live_df_features = pd.DataFrame(feature_vectors_list, columns=ORDERED_FEATURE_NAMES)

    logger.info(
        f"Successfully created DataFrame with {len(live_df_features)} feature vectors."
    )
    return live_df_features


if __name__ == "__main__":
    # Example usage of the new function
    feature_df = get_live_feature_vectors_from_ryu(num_samples_to_fetch=2)
    if not feature_df.empty:
        print("Feature DataFrame from Ryu:")
        print(feature_df.to_string())
    else:
        print("No feature DataFrame received or an error occurred.")
