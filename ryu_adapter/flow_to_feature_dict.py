import ipaddress
import time

ORDERED_FEATURE_NAMES = [
    'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
    'Timestamp', 'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s'
]

def ryu_flow_to_feature_dict(ryu_flow: dict) -> dict:
    features = {name: 0.0 for name in ORDERED_FEATURE_NAMES}
    match = ryu_flow.get('match', {})
    duration_sec = float(ryu_flow.get('duration_sec', 0))
    duration_nsec = float(ryu_flow.get('duration_nsec', 0))
    packet_count = float(ryu_flow.get('packet_count', 0))
    byte_count = float(ryu_flow.get('byte_count', 0))
    duration = duration_sec + (duration_nsec / 1_000_000_000)

    # Normalize OpenFlow 1.0 and 1.3 match fields
    ipv4_src = match.get('ipv4_src') or match.get('nw_src')
    ipv4_dst = match.get('ipv4_dst') or match.get('nw_dst')
    ip_proto = match.get('ip_proto') or match.get('nw_proto')
    tcp_src = match.get('tcp_src') or match.get('tp_src')
    tcp_dst = match.get('tcp_dst') or match.get('tp_dst')
    udp_src = match.get('udp_src') or match.get('tp_src')
    udp_dst = match.get('udp_dst') or match.get('tp_dst')

    try:
        features['Src IP'] = float(int(ipaddress.IPv4Address(ipv4_src)))
    except Exception:
        features['Src IP'] = 0.0
    try:
        features['Dst IP'] = float(int(ipaddress.IPv4Address(ipv4_dst)))
    except Exception:
        features['Dst IP'] = 0.0

    features['Src Port'] = float(tcp_src or udp_src or 0)
    features['Dst Port'] = float(tcp_dst or udp_dst or 0)
    features['Protocol'] = float(ip_proto or 0)
    features['Timestamp'] = time.time()
    features['Flow Duration'] = duration * 1_000_000  # microseconds

    if duration > 0:
        features['Flow Bytes/s'] = byte_count / duration
        features['Flow Packets/s'] = packet_count / duration

    return features
