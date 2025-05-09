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
from io import BytesIO
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt


def preview_data():
    """
        shows you the first 5 images in the train data
    """
    try:
        print("Azure Blob Storage Python quickstart sample")
        
        # path to low qual data
        LOW_RES = TRAIN_PATH + RES_OPTIONS[300]

        blob_data = container_client.list_blobs(name_starts_with=LOW_RES)

        # Get only the first 5 blobs
        for item in islice(blob_data, 5):
            image_data = get_bytestream(item)

            # Read the image using PIL (if it's an image)
            img = Image.open(image_data)
            plt.figure(figsize = (2,2))
            plt.title(f'{item.name[len(LOW_RES):]}')
            plt.imshow(img)

    except Exception as ex:
        print('Exception:')
        print(ex)
    
    return None


# return bytestream of some desired data (originally ItemPaged Class from a blob lookup)
def get_bytestream(item):
    """
    item
        ItemPaged class
    """
    blob_name = item.name

    # Create a BlobClient for the specific file
    blob_client = container_client.get_blob_client(blob_name)
    
    # Stream the blob directly into memory (not downloaded to disk)
    download_stream = blob_client.download_blob()
    
    # Use a BytesIO object to handle the image (or file) in memory
    image_data = BytesIO(download_stream.readall())
    return image_data

# return pandas dataframe with tabular metadata based on type wanted
def csv_data(data):
    """
        data: str
            you want train, 
                     test, or 
                     val?

        types must be compatible with metadata_types in constants.py
    """
    df = pd.DataFrame()

    try:
        if (data.strip().lower() in metadata_types):
            key = data
            val = metadata_types.get(key)

            blob_data = container_client.list_blobs(name_starts_with=val)
            for blob in blob_data:
                metadata = get_bytestream(blob)

                df = pd.read_csv(metadata)
    except Exception as ex:
        print('Exception:')
        print(ex)

    return df