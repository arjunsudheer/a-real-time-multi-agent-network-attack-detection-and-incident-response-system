import pyshark
import pandas as pd


def extract_features_from_pcap(pcap_file: str) -> pd.DataFrame:
    cap = pyshark.FileCapture(pcap_file)
    packets_data = []

    # Process all packets
    for packet in cap:
        packet_info = {}
        for layer in packet.layers:
            for field_name in layer.field_names:
                packet_info[field_name] = getattr(layer, field_name, None)
        packets_data.append(packet_info)

    cap.close()

    # Convert the list of packet data to a DataFrame
    df = pd.DataFrame(packets_data)
    return df


if __name__ == "__main__":
    pcap_path = "datasets/aci_iot_network_dataset_2023/original_dataset/pcap_combined/Combined_Pcaps/Benign Pcaps/2023-10-31-08_59_06_wireless.pcap"
    df = extract_features_from_pcap(pcap_path)
