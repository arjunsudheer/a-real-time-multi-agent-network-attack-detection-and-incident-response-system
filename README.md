# cyberwarrior-llm-challenge

This repository is dedicated to our project on LLM-based network attack detection and response for the CyberWarrior 2025 LLM Challenge.

Please follow this guide to run our implementation. The code for the demo is on the **main** branch. We use the Gemini API.

## Dataset

We use the ACI IOT Dataset for our experiment. Originally from: [https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle).

## File download Google Drive

Google Drive link: [https://drive.google.com/drive/folders/1MEbJYqekjPWPYNwZGwWgy40v0Wd7TE-0?usp=sharing](https://drive.google.com/drive/folders/1MEbJYqekjPWPYNwZGwWgy40v0Wd7TE-0?usp=sharing)


## Environment Setup

Create a python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate # on unix
pip install -r requirements.txt

```


Downlaod `label_encoder.pkl`, `test.csv`, and `scaler.pkl`, put them in project root


Download the classifier pre-trained weights `saved_models.zip` and extract them as `saved_models/` folder

```
unzip saved_models.zip
```

## SDN Simulation Setup: Ryu + Mininet + LLM-based Threat Detection

This guide documents how to set up a Software-Defined Network (SDN) simulation using **Mininet** and the **Ryu controller**, simulate attack traffic (e.g., SSH brute force), and feed that into a machine learning + LLM pipeline for threat detection and automated mitigation.

---

## Prerequisites

### On macOS (Host)

- Python 3.9 (use `pyenv`)
- Ryu SDN controller

Install Ryu in Python 3.9 virtualenv:

```bash
pyenv install 3.9.18
pyenv virtualenv 3.9.18 ryu39
pyenv activate ryu39

pip install ryu eventlet==0.30.2
```

>Ryu does not work with Python 3.12+. Use Python 3.9 for compatibility.

### On Ubuntu VM (Guest)

- Ubuntu running via UTM (or any hypervisor)

- Python 3.x

- Mininet, Open vSwitch

- Tools: hping3, nmap (optional for traffic simulation)

Install:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch hping3 nmap
```

### System Architecture (Mininet + RYU)
[macOS]
└── Ryu Controller (IP: 192.168.64.1)

[Ubuntu VM]
└── Mininet with:
    ├── h1, h2, h3 (virtual hosts)
    └── s1, s2, s3 (OpenFlow switches)

- Mininet creates a virtual network inside the VM.

- Ryu listens for switch traffic on macOS via REST and OpenFlow.

- The LLM pipeline runs on macOS and consumes classified flow data.


## Setup Instructions
### 1. Start Ryu Controller (on macOS)

In your terminal (inside the ryu39 environment):


```bash
ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest

```
This starts:

- A basic OpenFlow 1.3 switch controller
- A REST API on http://localhost:8080


### 2. Start Mininet (on Ubuntu VM)

Check Mac's IP from the VM:

```bash
ip route
```
Look for something like 192.168.64.1.

Launch Mininet:

```
sudo mn --controller=remote,ip=192.168.64.1 --topo=linear,3 --mac

```
This creates:

- 3 hosts: h1, h2, h3

- 3 switches: s1, s2, s3

- Automatically assigns MAC addresses

### 3.  Test Connectivity

In the Mininet CLI:

>mininet> pingall

Output:
```
*** 
Results: 0% dropped (6/6 received)
```

This confirms network is working and Ryu is handling switch logic.


### 4. Monitor Ryu Controller Logs
In the Ryu terminal (macOS), you'll see:

```
packet in 1 00:00:00:00:00:01 00:00:00:00:00:02 2
```
Meaning:
> Switch 1 received a packet from h1 → h2 and asked Ryu what to do with it.


Inspect flow entries:

> curl http://localhost:8080/stats/flow/1


## Running the Demo

To run the demo as shown in our video, use the following command:

```
python agents/network_agent_demo.py

```

## If the browser does not open automatically

If the browser does not open automatically, you can open it manually by going to the following link:

```
http://localhost:8000/index.html
```