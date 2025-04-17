# cyberwarrior-llm-challenge
This repository is dedicated to our project on LLM-based network attack detection and response for the CyberWarrior 2025 LLM Challenge.

Please follow this guide to run our implementation. The code for the demo is on the **main** branch. We use the Gemini API.

## Dataset
We use the ACI IOT Dataset for our experiment. Originally from: [https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023?select=ACI-IoT-2023_Kaggle).

After downloading the dataset and extracting the zip file, move all the extracted contents into a directory called "original_datasets". Your directory structure should look like the following:

- datasets
    - aci_iot_network_dataset_2023
        - original_dataset
            - <The extracted dataset files here>
- agents
- classifiers
- etc.

## File donwload Google Drive

Google Drive link: [https://drive.google.com/drive/folders/1MEbJYqekjPWPYNwZGwWgy40v0Wd7TE-0?usp=sharing]


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

## Running the Demo

To run the demo as shown in our video, use the following command:

```
python agents/network_aget_demo.py

```

