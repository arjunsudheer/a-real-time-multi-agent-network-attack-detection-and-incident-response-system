import ipaddress
import time

# IMPORTANT: USER MUST DEFINE THIS LIST ACCURATELY
# This list must contain the feature names in the exact order
# they appear in the data used to train the classifier models,
# excluding the 'Label' column.
ORDERED_FEATURE_NAMES = [
    "Dst IP",
    "Protocol",
    "Flow Bytes/s",
    "Timestamp",
    "Src IP",
    "Dst Port",
    "Flow Duration",
    "Flow Packets/s",
    "Src Port",
]

EXPECTED_FEATURE_COUNT = 9  # Based on the available data from mininet
if len(ORDERED_FEATURE_NAMES) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"ORDERED_FEATURE_NAMES must contain exactly {EXPECTED_FEATURE_COUNT} features, "
        f"but found {len(ORDERED_FEATURE_NAMES)}"
    )


def ryu_flow_to_feature_dict(ryu_flow: dict) -> dict:
    """
    Converts a Ryu flow entry dictionary into a feature dictionary.
    Many features will be defaulted as Ryu flow stats are not as rich as typical IDS datasets.
    The output features are raw, unscaled values.
    """
    # Initialize all features to 0.0 or an appropriate default
    # This ensures all features are present.
    features = {name: 0.0 for name in ORDERED_FEATURE_NAMES}

    match = ryu_flow.get("match", {})
    # Extract basic flow information from Ryu entry
    duration_sec = float(ryu_flow.get("duration_sec", 0))
    duration_nsec = float(ryu_flow.get("duration_nsec", 0))
    packet_count = float(ryu_flow.get("packet_count", 0))
    byte_count = float(ryu_flow.get("byte_count", 0))

    # --- Populate features that can be directly mapped or derived ---

    # Basic 5-tuple
    try:
        features["Src IP"] = float(
            int(ipaddress.IPv4Address(match.get("ipv4_src", "0.0.0.0")))
        )
    except ipaddress.AddressValueError:
        features["Src IP"] = 0.0
    features["Src Port"] = float(match.get("tcp_src", match.get("udp_src", 0)))

    try:
        features["Dst IP"] = float(
            int(ipaddress.IPv4Address(match.get("ipv4_dst", "0.0.0.0")))
        )
    except ipaddress.AddressValueError:
        features["Dst IP"] = 0.0
    features["Dst Port"] = float(match.get("tcp_dst", match.get("udp_dst", 0)))

    features["Protocol"] = float(match.get("ip_proto", 0))

    # Flow Duration in microseconds (as often seen in datasets)
    flow_duration_seconds = duration_sec + (duration_nsec / 1_000_000_000)
    features["Flow Duration"] = flow_duration_seconds * 1_000_000  # to microseconds

    # Timestamp (current time as epoch float)
    features["Timestamp"] = time.time()

    # Packet and Byte counts
    # Ryu doesn't distinguish Fwd/Bwd for a single flow entry easily.
    # We'll assign total counts to Fwd and 0 to Bwd, or split them if you have a heuristic.
    features["Total Fwd Packet"] = packet_count
    features["Total Length of Fwd Packet"] = byte_count

    # Rates
    if flow_duration_seconds > 0:
        features["Flow Bytes/s"] = byte_count / flow_duration_seconds
        features["Flow Packets/s"] = packet_count / flow_duration_seconds
        features["Fwd Packets/s"] = (
            packet_count / flow_duration_seconds
        )  # Assuming all are fwd

    else:
        features["Flow Bytes/s"] = 0.0
        features["Flow Packets/s"] = 0.0
        features["Fwd Packets/s"] = 0.0

    return features
