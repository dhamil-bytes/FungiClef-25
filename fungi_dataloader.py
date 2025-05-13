""""
Diane Hamilton

Multimodal data loader for fungi x csv data
"""

from constants import *
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import load_data
import torch
import matplotlib.pyplot as plt
import random


class MultimodalDataset(Dataset):
    def __init__(self, kind='train', res=300, transform=v2.Compose([v2.ToImage(),v2.Resize((224, 224)),v2.ToDtype(torch.float32, scale=True)])):
        """
        initializes batching given a dataframe of metadata and 
        """
        # get and clean data
        self.kind = kind
        self.df = load_data.csv_data(kind)
        self.df = load_data.standardize_data(self.df)

        # get path to img and csv data
        self.img_dir = PATH_OPTIONS.get(kind)
        self.res = RES_OPTIONS.get(res)
        self.img_path = self.img_dir + self.res
        self.meta_dir = metadata_types.get(kind)

        # initialize desired data augmentation
        self.transform = transform

        # labels and filenames (for batching)
        self.labels = self.df['category_id'] if kind in ['train','val'] else None
        self.filenames = self.df['filename']

        # drop the columns from the df
        self.df = self.df.drop(columns=['category_id'],axis=1) if kind in ['train', 'val'] else self.df
        self.df = self.df.drop(columns=['filename'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get csv row
        # tabular = self.df.iloc[idx]
        filename = self.filenames.iloc[idx]
        tabular = self.df.iloc[idx].values

        
        # get img and apply img transform
        img_path = os.path.join(self.img_path, filename)
        # get from the blob and turn to img
        blob_data = container_client.list_blobs(name_starts_with=img_path)
        for item in blob_data:
            image = load_data.get_bytestream(item)
            image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.kind in ['train', 'val']:
            label = self.labels.iloc[idx]
            return image, torch.tensor(tabular,dtype=torch.float32), torch.tensor(label,dtype=torch.int64)
    
        return image, torch.tensor(tabular,dtype=torch.float32)

    def visualize_data(self, n=5):
        rndm_n = random.sample(range(len(self)), n)

        plt.figure(figsize=(4 * n, 4))  # Widened layout for better spacing
        for i in range(n):
            k = rndm_n[i]
            sample = self.__getitem__(k)

            if self.kind in ['train', 'val']:
                img, _, label = sample
                label_text = f'Category: {label.item()}'
            else:
                img, _ = sample
                label_text = 'Category: UNK'

            img = img.permute(1, 2, 0).numpy()  # C x H x W → H x W x C

            plt.subplot(1, n, i + 1)
            plt.imshow(img)
            plt.title(label_text)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    def download_dataset(self):
        self.df.to_csv(f'{self.kind}_multimodal_data.csv',index=False)