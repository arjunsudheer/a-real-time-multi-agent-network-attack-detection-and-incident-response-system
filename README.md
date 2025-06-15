# cyberwarrior-llm-challenge

This repository is dedicated to our project on LLM-based network attack detection and response for the CyberWarrior 2025 LLM Challenge.

Please follow this guide to run our implementation. The code for the demo is on the **main** branch. We use the Gemini API.

## Video Demo Link

[https://youtu.be/CRR4-pxZ01Q](https://youtu.be/CRR4-pxZ01Q)

## Dataset

We use the ACI IOT Dataset for our experiment. Originally from: [https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle).

## File download Google Drive

Please download the datasets.zip file from the Google Drive Link provided below. Once you have downloaded the zip file, unzip it in the project root directory using the following command:

```
unzip datasets.zip
```

Google Drive link: [https://drive.google.com/drive/folders/1MEbJYqekjPWPYNwZGwWgy40v0Wd7TE-0?usp=sharing](https://drive.google.com/drive/folders/1MEbJYqekjPWPYNwZGwWgy40v0Wd7TE-0?usp=sharing)

## Environment Setup

Create a python virtual environment

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## SDN Simulation Setup: Ryu + Mininet + LLM-based Threat Detection

This section documents how to set up a Software-Defined Network (SDN) simulation using **Mininet** and the **Ryu controller**, simulate attack traffic (e.g., SSH brute force), and feed that into a machine learning + LLM pipeline for threat detection and automated mitigation.

---

### On Your Host Machine

- Python 3.9 (use `pyenv`)
- Ryu SDN controller

Install Ryu in Python 3.9 virtualenv:

```
pyenv install 3.9.18
pyenv virtualenv 3.9.18 ryu39
pyenv activate ryu39

pip3 install ryu eventlet==0.30.2
```

> Ryu does not work with Python 3.12+. Use Python 3.9 for compatibility.

### On Ubuntu VM (Guest)

- Ubuntu running via UTM (or any hypervisor)

- Python 3.x

- Mininet, Open vSwitch

- Tools: hping3, nmap (optional for traffic simulation)

Install:

```
sudo apt update
sudo apt install -y mininet openvswitch-switch hping3 nmap
```

### System Architecture (Mininet + RYU)

[Host]

└── Ryu Controller (IP: <Your Host Machine IP Address>)

[Ubuntu VM]

└── Mininet with:

└── h1, h2 (virtual hosts)

└── s1, s2 (OpenFlow switches)

- Mininet creates a virtual network inside the VM.

- Ryu listens for switch traffic on Host via REST and OpenFlow.

- The LLM pipeline runs on Host and consumes classified flow data.

## Setup Instructions

### 1. Start Ryu Controller (Host)

In your terminal (inside the ryu39 environment):

```
ryu-manager ryu_adapter/simple_switch_13_custom.py ryu.app.ofctl_rest
```

This starts:

- A basic OpenFlow 1.3 switch controller
- A REST API on http://localhost:8080

### 2. Start Mininet (Ubuntu VM)

Check your host machine's IP from the VM:

```
ip route
```

Look for something like 192.168.64.1.

Launch Mininet:

```
sudo mn --controller=remote,ip=<Your Host Machine IP Address> --topo=linear,2 --mac
```

This creates:

- 2 hosts: h1, h2

- 2 switches: s1, s2

- Automatically assigns MAC addresses

### 3. Test Connectivity

In the Mininet CLI:

> mininet> pingall

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

## Simulate network attacks from mininet

Run these commands in your Ubuntu VM running mininet. After you run the command, return to your host machine to run the demo, which we explain how to do in the next section.

> Just a quick note: To simulate different attacks from mininet, you should clear the mininet cache using this command in your terminal. If you are still in the mininet interactive shell please exit before running the following command:

```
sudo mn -c
```

### Dictionary Attack (SSH Brute Force)

```
h1 hping3 -S -p 22 -i u1000 10.0.0.2
```

### DNS Flood

```
h1 hping3 --udp -p 53 -i u1000 10.0.0.2
```

### Benign (Legit TCP Connection)

```
h1 telnet 10.0.0.2 80
```

## Running the Demo

To run the demo as shown in our video, use the following command:

```
python3 -m network_agent_system
```

### If the browser does not open automatically

If the browser does not open automatically, you can open it manually by going to the following link:

```
http://localhost:8000/index.html
```
