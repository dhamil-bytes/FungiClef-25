"""
Diane Hamilton

load_data.py
    Load data from my azure storage container
"""

import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from constants import *
from itertools import islice

try:
    print("Azure Blob Storage Python quickstart sample")

    # path to low qual data
    LOW_RES = TRAIN_PATH + RES_OPTIONS[300]

    blob_data = container_client.list_blobs(name_starts_with=LOW_RES)
    print(LOW_RES)
    print(blob_data)

    # Get only the first 5 blobs
    for item in islice(blob_data, 5):
        print("File:", item.name)


except Exception as ex:
    print('Exception:')
    print(ex)