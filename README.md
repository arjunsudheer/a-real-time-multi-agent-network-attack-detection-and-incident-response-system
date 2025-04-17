# cyberwarrior-llm-challenge
This repository is dedicated to our project on LLM-based network attack detection and response for the CyberWarrior 2025 LLM Challenge.

Please follow this guide to run our implementation. The code for the demo is on the **main** branch. We use the Gemini API.

## Dataset
We use the ACI IOT Dataset for our experiment. You can download the dataset at this link: [https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle).

After downloading the dataset and extracting the zip file, move all the extracted contents into a directory called "original_datasets". Your directory structure should look like the following:

- datasets
    - aci_iot_network_dataset_2023
        - original_dataset
            - <The extracted dataset files here>
- agents
- classifiers
- etc.

## Feature Selection Agent (Data preprocessing)
To preprocess the data, run the following commands: