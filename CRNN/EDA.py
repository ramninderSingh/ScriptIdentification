from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image,UnidentifiedImageError
import random
import re

class ImageDataset1(Dataset):
    def __init__(self, root_dir, label, max_images=200000, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.images = []

        for folder_1 in os.listdir(root_dir):
            folder_1_path = os.path.join(root_dir, folder_1)
            if os.path.isdir(folder_1_path):
                for folder_2 in os.listdir(folder_1_path):
                    folder_2_path = os.path.join(folder_1_path, folder_2)
                    if os.path.isdir(folder_2_path):
                        for image_file in os.listdir(folder_2_path):
                            image_path = os.path.join(folder_2_path, image_file)
                            self.images.append(image_path)

        if len(self.images) > max_images:
            self.images = random.sample(self.images, max_images)

        self.labels = [label] * len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        filename = image_path.split('/')[-1]
        cleaned_name = re.sub(r'[\d_]', '', filename)
        self.word_length = len(cleaned_name)-4
        try:
            with open(image_path, 'rb') as f:
                _ = f.read(1)
            image = Image.open(image_path)
            # image = image.resize((128, 64))  
            label = self.labels[idx]

            if self.transform is not None:
                image = self.transform(image)

            return image, label
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error reading file '{image_path}': {str(e)}")
            # Skip this file and choose another
            return self.__getitem__(np.random.randint(len(self)))  # Retry with a random index


class ImageDataset2(Dataset):
    def __init__(self, root_dir, label, max_images=400000, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.images = []

        for image_file in os.listdir(root_dir):
            image_path = os.path.join(root_dir, image_file)
            self.images.append(image_path)

        if len(self.images) > max_images:
            self.images = random.sample(self.images, max_images)

        self.labels = [label] * len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        filename = image_path.split('/')[-1]
        cleaned_name = re.sub(r'[\d_]', '', filename)
        self.word_length = len(cleaned_name)-4
        try:
            image = Image.open(image_path)
             # Resize the image width height
            # image = image.resize((128, 64)) 
            label = self.labels[idx]
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            return image,label

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))  # Retry with a random index
            
        

        return image, label





class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmentations=None, post_transforms=None):
        self.dataset = dataset
        self.augmentations = augmentations
        self.post_transforms = post_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.augmentations is not None:
            image = self.augmentations(image)
        if self.post_transforms is not None:
            image = self.post_transforms(image)
        return image, label