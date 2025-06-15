#!/usr/bin/env python3
"""
Test script for the simplified ResponseAgent
Demonstrates how to generate and optionally execute mitigation commands
"""

from agents.response_agent import ResponseAgent
import pandas as pd


def test_response_agent():
    # Initialize the response agent
    agent = ResponseAgent()

    print("=== ResponseAgent Test Suite ===\n")

    # Test 1: Block all traffic from h1 (10.0.0.1)
    print("Test 1: Block All Traffic from Host h1 (10.0.0.1)")
    result1 = agent.generate_and_run_mitigation_commands(
        src_ip="10.0.0.1", dpid=1, execute=False  # Set to True to actually execute
    )
    print(result1["summary"])
    print(f"Generated command: {result1['commands'][0].curl_command}\n")

    # Test 2: Block TCP SYN Flood to Port 22
    print("Test 2: Block TCP SYN Flood to Port 22")
    result2 = agent.generate_and_run_mitigation_commands(
        src_ip="10.0.0.1", protocol="tcp", dst_port=22, dpid=1, execute=False
    )
    print(result2["summary"])
    print(f"Generated command: {result2['commands'][0].curl_command}\n")

    # Test 3: Block ICMP (Ping) from h1
    print("Test 3: Block ICMP (Ping) from h1")
    result3 = agent.generate_and_run_mitigation_commands(
        src_ip="10.0.0.1", protocol="icmp", dpid=1, execute=False
    )
    print(result3["summary"])
    print(f"Generated command: {result3['commands'][0].curl_command}\n")

    # Test 4: Block UDP Flood from h1 to port 80
    print("Test 4: Block UDP Flood from h1 to port 80")
    result4 = agent.generate_and_run_mitigation_commands(
        src_ip="10.0.0.1", protocol="udp", dst_port=80, dpid=1, execute=False
    )
    print(result4["summary"])
    print(f"Generated command: {result4['commands'][0].curl_command}\n")

    # Test 5: Test with classification results (backward compatibility)
    print("Test 5: Using classification results format")

    # Mock classification results
    classification_results = {
        "final_prediction": "DDoS Attack",
        "pre_detection": {"prediction": "Malicious", "confidence": 90.0},
    }

    # Mock original sample
    original_sample = pd.DataFrame(
        {
            "Src IP": ["10.0.0.2"],
            "Dst IP": ["10.0.0.3"],
            "Dst Port": [443],
            "Protocol": [6],  # TCP
        }
    )

    commands = agent.generate_mitigation_commands(
        classification_results=classification_results,
        original_sample=original_sample,
        dpid=2,
    )

    summary = agent.get_mitigation_summary(commands)
    print(summary)
    print(f"Generated command: {commands[0].curl_command}\n")

    print("=== All Tests Completed ===")
    print(
        "Note: Set execute=True to actually run the curl commands against Ryu controller"
    )


if __name__ == "__main__":
    test_response_agent()
