import ipaddress
import time

# IMPORTANT: USER MUST DEFINE THIS LIST ACCURATELY
# This list must contain the feature names in the exact order
# they appear in the data used to train the classifier models,
# excluding the 'Label' column.
ORDERED_FEATURE_NAMES = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Fwd URG Flags', # Note: Bwd PSH/URG Flags are often not in basic flow stats
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    # Bulk features are typically harder to get from simple flow entries
    'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', # Assuming Fwd Bulk features are excluded based on previous discussions
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes', # Note: Subflow Bwd Packets was missing, added Subflow Bwd Bytes
    'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Connection Type'
]

EXPECTED_FEATURE_COUNT = 78 # Based on the provided header list minus 'Label'
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
    # This ensures all 78 features are present.
    features = {name: 0.0 for name in ORDERED_FEATURE_NAMES}

    match = ryu_flow.get('match', {})
    # Extract basic flow information from Ryu entry
    duration_sec = float(ryu_flow.get('duration_sec', 0))
    duration_nsec = float(ryu_flow.get('duration_nsec', 0))
    packet_count = float(ryu_flow.get('packet_count', 0))
    byte_count = float(ryu_flow.get('byte_count', 0))

    # --- Populate features that can be directly mapped or derived ---

    # Basic 5-tuple
    try:
        features['Src IP'] = float(int(ipaddress.IPv4Address(match.get('ipv4_src', '0.0.0.0'))))
    except ipaddress.AddressValueError:
        features['Src IP'] = 0.0
    features['Src Port'] = float(match.get('tcp_src', match.get('udp_src', 0)))

    try:
        features['Dst IP'] = float(int(ipaddress.IPv4Address(match.get('ipv4_dst', '0.0.0.0'))))
    except ipaddress.AddressValueError:
        features['Dst IP'] = 0.0
    features['Dst Port'] = float(match.get('tcp_dst', match.get('udp_dst', 0)))

    features['Protocol'] = float(match.get('ip_proto', 0))

    # Flow Duration in microseconds (as often seen in datasets)
    flow_duration_seconds = duration_sec + (duration_nsec / 1_000_000_000)
    features['Flow Duration'] = flow_duration_seconds * 1_000_000 # to microseconds

    # Timestamp (current time as epoch float)
    features['Timestamp'] = time.time()

    # Flow ID (defaulting to 0, as it's usually an identifier not for training)
    features['Flow ID'] = 0.0

    # Packet and Byte counts
    # Ryu doesn't distinguish Fwd/Bwd for a single flow entry easily.
    # We'll assign total counts to Fwd and 0 to Bwd, or split them if you have a heuristic.
    features['Total Fwd Packet'] = packet_count
    features['Total Bwd packets'] = 0.0 # Placeholder
    features['Total Length of Fwd Packet'] = byte_count
    features['Total Length of Bwd Packet'] = 0.0 # Placeholder

    # Subflow counts (approximated)
    features['Subflow Fwd Packets'] = features['Total Fwd Packet']
    features['Subflow Fwd Bytes'] = features['Total Length of Fwd Packet']
    features['Subflow Bwd Bytes'] = features['Total Length of Bwd Packet'] # Will be 0.0

    # Rates
    if flow_duration_seconds > 0:
        features['Flow Bytes/s'] = byte_count / flow_duration_seconds
        features['Flow Packets/s'] = packet_count / flow_duration_seconds
        features['Fwd Packets/s'] = packet_count / flow_duration_seconds # Assuming all are fwd
        features['Bwd Packets/s'] = 0.0
    else:
        features['Flow Bytes/s'] = 0.0
        features['Flow Packets/s'] = 0.0
        features['Fwd Packets/s'] = 0.0
        features['Bwd Packets/s'] = 0.0

    # Mean packet lengths (simplistic: total bytes / total packets)
    if packet_count > 0:
        avg_pkt_size = byte_count / packet_count
        features['Fwd Packet Length Mean'] = avg_pkt_size
        features['Fwd Packet Length Max'] = avg_pkt_size # Simplistic default
        features['Fwd Packet Length Min'] = avg_pkt_size # Simplistic default
        features['Average Packet Size'] = avg_pkt_size
        features['Packet Length Mean'] = avg_pkt_size # Assuming mostly fwd traffic
        features['Packet Length Max'] = avg_pkt_size # Simplistic
        features['Packet Length Min'] = avg_pkt_size # Simplistic
        features['Fwd Segment Size Avg'] = avg_pkt_size # Approximation
    else:
        features['Fwd Packet Length Mean'] = 0.0
        features['Fwd Packet Length Max'] = 0.0
        features['Fwd Packet Length Min'] = 0.0
        features['Average Packet Size'] = 0.0
        features['Packet Length Mean'] = 0.0
        features['Packet Length Max'] = 0.0
        features['Packet Length Min'] = 0.0
        features['Fwd Segment Size Avg'] = 0.0

    # Fwd Act Data Pkts (approximated as total fwd packets)
    features['Fwd Act Data Pkts'] = features['Total Fwd Packet']
    # Fwd Seg Size Min (approximated)
    if 'Fwd Packet Length Min' in features: # Check if already populated
        features['Fwd Seg Size Min'] = features['Fwd Packet Length Min']

    # --- Defaulting features that are hard to derive from Ryu's aggregated stats ---
    # Most IATs, detailed packet length stats (Std, Variance), TCP flags counts,
    # header lengths, Bwd stats, Bulk stats, Active/Idle stats, Window Bytes, Connection Type
    # are defaulted to 0.0 by the initial dictionary creation.
    # Specific defaults can be refined here if needed.

    # Example: Connection Type (could be mapped from protocol if desired)
    # For now, it defaults to 0.0.
    # if features['Protocol'] == 6: features['Connection Type'] = 1.0 # TCP
    # elif features['Protocol'] == 17: features['Connection Type'] = 2.0 # UDP

    # You **MUST** review and enhance this function.
    # For many of the 78 features, you'll need to decide on appropriate defaults
    # or more sophisticated ways to estimate them if possible.
    # The current version will produce a vector of the correct SHAPE but
    # its SEMANTIC meaning might be very different from your test.csv,
    # potentially leading to poor classification performance.

    return features
