import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, DataLoader, random_split, Dataset,ConcatDataset
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import random
from torchvision.datasets import ImageFolder
from PIL import Image,UnidentifiedImageError
from tempfile import TemporaryDirectory
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow
plt.ion()
import logging

mps_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {mps_device}")
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define data transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Ensure images have 3 channels
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Ensure images have 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# STRAugmentations class definition remains the same
class STRAugmentations:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.ops = [Curve(rng=self.rng), Rotate(rng=self.rng), Perspective(self.rng), Distort(self.rng), Stretch(self.rng), Shrink(self.rng), TranslateX(self.rng),
                    TranslateY(self.rng), VGrid(self.rng), HGrid(self.rng), Grid(self.rng), RectGrid(self.rng), EllipseGrid(self.rng)]
        self.ops.extend([GaussianNoise(self.rng),ShotNoise(self.rng), ImpulseNoise(self.rng), SpeckleNoise(self.rng)])
        self.ops.extend([GaussianBlur(self.rng),DefocusBlur(self.rng), MotionBlur(self.rng), ZoomBlur(self.rng)])
        self.ops.extend([Contrast(self.rng), Brightness(self.rng), JpegCompression(self.rng), Pixelate(self.rng)])
        self.ops.extend([Fog(self.rng), Snow(self.rng), Frost(self.rng), Rain(self.rng), Shadow(self.rng)])
        self.ops.extend([Posterize(self.rng), Solarize(self.rng), Invert(self.rng), Equalize(self.rng), AutoContrast(self.rng), Sharpness(self.rng), Color(self.rng)])

    def __call__(self, img):
        op = self.rng.choice(self.ops)
        mag = self.rng.integers(0, 3)
        return op(img, mag=mag)

# ImageDataset1 and ImageDataset2 classes definitions remain the same with slight modifications
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
        try:
            with open(image_path, 'rb') as f:
                _ = f.read(1)
            image = Image.open(image_path)
                # Resize the image
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
        try:
            image = Image.open(image_path)
            # image = image.resize((100, 32))  # Resize the image
            if image.mode == 'RGBA':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise e

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# # # Parameters
# max_images = 100000

# # Create datasets without augmentations initially
# gujarati_dataset = ImageDataset2('Guj1/train', label=1, max_images=max_images, transform=None)
# print("Created Guj")
# hindi_dataset = ImageDataset1('hindi', label=0, max_images=max_images, transform=None)
# print("Created Hindi")
# english_dataset = ImageDataset1('english', label=2, max_images=max_images, transform=None)
# print("Created English")

# # Concatenate datasets
# full_dataset = ConcatDataset([hindi_dataset, english_dataset, gujarati_dataset])
# print("Created full dataset")

# # Split into training and validation sets
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


# Parameters
max_images = 10000

hindi_dataset = ImageDataset2('recognition/train/hindi', label=0, max_images=max_images, transform=None)
english_dataset = ImageDataset2('recognition/train/english', label=1, max_images=max_images, transform=None)
gujarati_dataset = ImageDataset2('recognition/train/punjabi', label=2, max_images=max_images, transform=None)

# Combine datasets
full_dataset = ConcatDataset([hindi_dataset, english_dataset])
train_size = int(0.8 * len(full_dataset)) 
val_size = len(full_dataset) - train_size 
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])



# Apply augmentations only to the training dataset
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

# Apply augmentations and post-transforms to the training dataset
augmentations = STRAugmentations(seed=0)
train_dataset = AugmentedDataset(train_dataset, post_transforms=train_transforms)

# Apply only post-transforms to the validation dataset
val_dataset = AugmentedDataset(val_dataset, post_transforms=train_transforms)

batch_size = 512
num_workers = 8

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("Data loaders created")



# batch_size = 512
# num_workers = 8

# train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
# val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)



def train_model(model, train_loader, validation_loader, optimizer, criterion, n_epochs=20, early_stopping_patience=5):
    model = model.to(mps_device)
    train_cost_list = []
    val_cost_list = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(n_epochs):
        train_cost = 0
        logging.info(f"Training Epoch: {epoch + 1}")

        # Training phase
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as t:
            for x, y in t:
                x = x.to(mps_device)
                y = y.to(mps_device)
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                train_cost += loss.item()
                t.set_postfix(train_loss=train_cost / (len(t)))

        train_cost /= len(train_loader)
        train_cost_list.append(train_cost)

        # Validation phase
        model.eval()
        val_cost = 0
        correct = 0
        with tqdm(validation_loader, desc="Validation", unit="batch") as t:
            for x_test, y_test in t:
                x_test = x_test.to(mps_device)
                y_test = y_test.to(mps_device)
                z = model(x_test)
                loss = criterion(z, y_test)
                _, yhat = torch.max(z.data, 1)
                val_cost += loss.item()

        val_cost /= len(validation_loader)
        val_cost_list.append(val_cost)

        # Early stopping logic
        if val_cost < best_val_loss:
            best_val_loss = val_cost
            best_epoch = epoch
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logging.info(f'Early stopping after epoch {epoch + 1} as validation loss did not improve.')
                break

        logging.info("--> Epoch Number : {} | Training Loss : {:.4f} | Validation Loss : {:.4f} | Validation Accuracy : {:.2f}%".format(
            epoch + 1, train_cost, val_cost, (correct / len(validation_loader.dataset)) * 100
        ))

    return train_cost_list, val_cost_list, model



# Alexnet
alexnet = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
for param in alexnet.parameters():
    param.requires_grad = False

num_ftrs_alexnet = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Sequential(
    nn.Linear(num_ftrs_alexnet, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 3)
)
optimizer_conv_alexnet = optim.SGD(alexnet.classifier[6].parameters(), lr=0.001, momentum=0.9)




# # Resnet18
# resnet18 = models.resnet18(weights='IMAGENET1K_V1')

# for param in resnet18.parameters():
#     param.requires_grad = False


# num_ftrs_resnet18 = resnet18.fc.in_features
# resnet18.fc = nn.Sequential(
#     nn.Linear(num_ftrs_resnet18, 512),
#     nn.LeakyReLU(),
#     nn.Linear(512, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 3)
# )

# optimizer_conv_resnet = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)



# #Retrained Alexnet
# alexnet_1 = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
# for param in alexnet_1.parameters():
#     param.requires_grad = False

# num_ftrs_alexnet_1 = alexnet_1.classifier[6].in_features
# alexnet_1.classifier[6] = nn.Sequential(
#     nn.Linear(num_ftrs_alexnet_1, 512),
#     nn.LeakyReLU(),
#     nn.Linear(512, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 2)
# )
# optimizer_conv_alexnet_1 = optim.SGD(alexnet_1.classifier[6].parameters(), lr=0.001, momentum=0.9)
# alexnet_1.load_state_dict(torch.load("models/alexnet/alex_syn_2.pt"))

# #Retrained resnet18
# resnet18_1 = models.resnet18(weights='IMAGENET1K_V1')

# for param in resnet18_1.parameters():
#     param.requires_grad = False


# num_ftrs_resnet18_1 = resnet18_1.fc.in_features
# resnet18_1.fc = nn.Sequential(
#     nn.Linear(num_ftrs_resnet18_1, 512),
#     nn.LeakyReLU(),
#     nn.Linear(512, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 3)
# )
# optimizer_conv_resnet_1 = optim.SGD(resnet18_1.fc.parameters(), lr=0.001, momentum=0.9)
# resnet18_1.load_state_dict(torch.load("models/resnet/res_synaug_3.pt"))

criterion = nn.CrossEntropyLoss()
print("Starting Training")
train_cost_listv5, val_cost_listv5, model_to_save=train_model(model=alexnet, 
                                                                       train_loader=train_loader, 
                                                                       validation_loader=val_loader, 
                                                                       optimizer=optimizer_conv_alexnet,
                                                                       criterion = nn.CrossEntropyLoss(),
                                                                       n_epochs=100)

torch.save(model_to_save.state_dict(), "alex_real_HEP.pt")
logging.info("Training Complete. Model saved as alex_realaug_3.pt")