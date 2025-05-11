"""
Diane Hamilton

constants.py
    Holds all constant values for my program
    Hint: There are some things i simply can't share. 
          I'm sorry, Internet. 
          Though I'm sure you'll find the contents of this file in your systems.
          Someday... hopefully not soon!
"""


import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, BlobPrefix

account_url = "https://imgmushroom.blob.core.windows.net/"
container_name = "training-images"
main_folder = "FungiTastic-FewShot"
delimiter = "/"
blob_prefix = "fungiclef25" + delimiter
img_prefix = blob_prefix + "images" + delimiter
metadata_prefix = blob_prefix + "metadata" + delimiter
csv_prefix = metadata_prefix + main_folder + delimiter
data_prefix = img_prefix + main_folder + delimiter

TRAIN_PATH = data_prefix + "train" + delimiter
TEST_PATH = data_prefix + "test" + delimiter
VAL_PATH = data_prefix + "val" + delimiter

PATH_OPTIONS = {'train': TRAIN_PATH,
                "test": TEST_PATH,
                "val": VAL_PATH}

RES_OPTIONS = {300: "300p" + delimiter,
               500: "500p" + delimiter,
               720: "720p" + delimiter,
               "full": "fullsize" + delimiter}

metadata_types = {"train": csv_prefix + main_folder + "-Train.csv",
                  "test": csv_prefix + main_folder + "-Test.csv",
                  "val": csv_prefix + main_folder + "-Val.csv"}

default_credential = DefaultAzureCredential()

try:
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    # Get container client
    container_client = blob_service_client.get_container_client(container_name)

except Exception as ex:
    print('Exception:')
    print(ex)