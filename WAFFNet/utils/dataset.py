import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

STANDARD_SIZE = (224, 224)

def build_dataset_index(base_path, folders, save_path):
    data = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for root, _, files in os.walk(folder_path):
            for filename in tqdm(files, desc=f"Collecting {folder}"):
                if filename.lower().endswith(valid_exts):
                    image_path = os.path.join(root, filename)
                    data.append({'image_path': image_path, 'label': folder})

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)
    print(f" Dataset index built with {len(df)} samples. Saved at {save_path}")
    return df

class RecognitionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path, label = row['image_path'], row['label']

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)

        label_idx = self.label2idx[label]
        return img, label_idx
    
class ImageData(Dataset):
    def __init__(self, dataframe, transform=None, subcategories=None):
        if isinstance(dataframe, torch.utils.data.Subset):
            self.data = dataframe.dataset.iloc[dataframe.indices].reset_index(drop=True)
        else:
            self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.subcategories = subcategories

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label_name = row['label']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new("RGB", STANDARD_SIZE, color="gray")

        if self.transform:
            image = self.transform(image)

        label = self.subcategories.index(label_name)
        return image, label 

class TestDataset(Dataset):
    def __init__(self, dataframe, transform , subcategories):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.subcategories = subcategories
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        label_name = row['label']
        filename = os.path.basename(img_path)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        label = self.subcategories.index(label_name)
        return img, label, filename
    
def create_dataloaders(df, train_transforms, val_transforms, subcategories, batch_size=32):
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_df, val_df = random_split(df, [train_size, val_size])

    train_loader = DataLoader(
        ImageData(train_df, transform=train_transforms, subcategories=subcategories),
        batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        ImageData(val_df, transform=val_transforms, subcategories=subcategories),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader

def create_test_loader(dataframe, test_transform, subcategories, batch_size=32):
    test_dataset = TestDataset(dataframe, test_transform, subcategories)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader