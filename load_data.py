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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


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

# bar graph to visualize catagory type column distributions
def visualize_categories(data):
    """
    data: str
        you want train, 
                    test, or 
                    val?

    types must be compatible with metadata_types in constants.py
    """

    embedding_targets = ['order','family','genus','specificEpithet','region','district']

    if (data.strip().lower() in metadata_types):
        df = csv_data(data)

        le = LabelEncoder()
        cols = df.columns
        for c in cols:
            has_na = df[c].isna()
            if c in embedding_targets or c == 'category_id':
                if has_na.sum() != 0:
                    # fill as UNK
                    df.loc[has_na,c] = 'UNK'
                df[c] = df[c].astype('category')
                vis_col = le.fit_transform(df[c])
                vis_col = pd.Series(vis_col)
                value_counts = vis_col.value_counts()
                plt.figure(figsize=(8, 6))
                value_counts.plot(kind='bar')
                plt.title(f'Frequency of Categories in {c}')
                plt.xlabel('Category')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()


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

# receives a dataframe and standardizes all float64 data with standardscalar
# also turns eventDate into datetime object
def standardize_data(data: pd.DataFrame):
    """
    data
        a pandas dataframe
    """
    # drop redundant data
    if 'eventDate' in data.columns:
        data = data.drop(columns=['eventDate'],  axis=1)
    if 'scientificName' in data.columns:
        data = data.drop(columns=['scientificName'], axis=1)
    if 'species' in data.columns:
        data = data.drop(columns=['species'], axis=1)
    if 'observationID' in data.columns:
        data = data.drop(columns=['observationID'], axis=1)

    # get int64 and binarize that ho
    norml_targets = ['coorUncert','elevation','landcover']
    embedding_targets = ['order','family','genus','specificEpithet','region','district']
    onehot_targets = ['biogeographicalRegion', 'metaSubstrate', 'substrate', 'iucnRedListCategory', 'class', 'phylum', 'kingdom', 'countryCode', 'habitat','poisonous','hasCoordinate']
    for c in data.columns:
        le = LabelEncoder()
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # check for unknown vals
        has_na = data[c].isna()

        # normalize using standardscalar
        if c in norml_targets:
            if has_na.sum() != 0:
                # picking the median for potentially skewed data
                data.loc[has_na, c] = data[c].median()
            data[c] = scaler.fit_transform(X=data[[c]])

        # onehot encode
        elif c in onehot_targets:
            if has_na.sum() != 0:
                # fill as UNK
                data.loc[has_na, c] = 'UNK'
            data[c] = data[c].astype('category')
            
            encoded = encoder.fit_transform(data[[c]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([c]))
            encoded_df.index = data.index
            data = pd.concat([data.drop(columns=[c]), encoded_df], axis=1)
        
        # label encode
        elif c in embedding_targets:
            if has_na.sum() != 0:
                # fill as UNK
                data.loc[has_na, c] = 'UNK'
            data[c] = data[c].astype('category')
            data[c] = le.fit_transform(data[c])
    
    # apply astype(float) for consistent dataloading
    for c in data.columns:
        if c != 'filename':
            try:
                data[c] = data[c].astype(float)
            except Exception as x:
                print(x)
                data.to_csv('failed_standardize_data.csv')
                raise(x)

    # no col for index
    data = data.reset_index(drop=True)
    return data