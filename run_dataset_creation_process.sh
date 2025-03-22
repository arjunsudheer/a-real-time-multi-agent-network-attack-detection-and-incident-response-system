#!/bin/bash

cd datasets/nsl_kdd/
python3 create_clean_dataset.py

cd ../aci_iot_network_dataset_2023/
python3 create_clean_dataset.py

cd ../cic_iot_dataset_2023/
python3 create_clean_dataset.py
