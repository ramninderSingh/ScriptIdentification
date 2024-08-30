import os
import random
import logging
import warnings
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import torchvision.datasets as datasets

warnings.filterwarnings("ignore")

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up logging
log_file = 'testing_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),  # Ensure images have 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, root_dir, label, max_images=400000, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.images = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

        if len(self.images) > max_images:
            self.images = random.sample(self.images, max_images)

        self.labels = [label] * len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            logging.error(f"Image type is {type(image)}. Expected torch.Tensor.")

        return image, label

# Parameters
max_images = 10000

# Create datasets
hindi_dataset = ImageDataset('ScriptDataset/TestDataset/BSTD/hindi', label=0, max_images=max_images, transform=data_transforms)
english_dataset = ImageDataset('ScriptDataset/TestDataset/BSTD/english', label=1, max_images=max_images, transform=data_transforms)
gujarati_dataset = ImageDataset('ScriptDataset/TestDataset/BSTD/gujarati', label=2, max_images=max_images, transform=data_transforms)
hindi_dataset1 = ImageDataset('ScriptDataset/TrainDataset/BSTD/hindi', label=0, max_images=max_images, transform=data_transforms)
english_dataset1 = ImageDataset('ScriptDataset/TrainDataset/BSTD/english', label=1, max_images=max_images, transform=data_transforms)
gujarati_dataset1 = ImageDataset('ScriptDataset/TrainDataset/BSTD/gujarati', label=2, max_images=max_images, transform=data_transforms)

# Combine datasets
full_dataset = ConcatDataset([hindi_dataset, english_dataset,gujarati_dataset])

# Create a DataLoader
batch_size = 512
num_workers = 8
test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print("Created Full_Dataset")

def test_model(model, validation_loader):
    correct = 0
    total = 0
    all_y_true = []
    all_y_pred = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        with tqdm(validation_loader, desc="Testing", unit="batch") as t:
            for x_test, y_test in t:
                x_test, y_test = x_test.to(device), y_test.to(device)
                outputs = model(x_test)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_test).sum().item()
                total += y_test.size(0)
                all_y_true.extend(y_test.cpu().numpy())
                all_y_pred.extend(preds.cpu().numpy())

    accuracy = correct / total
    f1_weighted = f1_score(all_y_true, all_y_pred, average='weighted')
    class_report = classification_report(all_y_true, all_y_pred, output_dict=True)

    overall_precision = class_report['macro avg']['precision'] * 100
    overall_recall = class_report['macro avg']['recall'] * 100
    overall_f1 = class_report['macro avg']['f1-score'] * 100
    logging.info("Classification Report:\n%s", classification_report(all_y_true, all_y_pred))
    logging.info(f"Testing Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Overall Precision: {overall_precision:.2f}%")
    logging.info(f"Overall Recall: {overall_recall:.2f}%")
    logging.info(f"Overall F1 Score: {overall_f1:.2f}%")

    return accuracy

# Load model and modify for our use case
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

model_name = "models/alexnet/alex_realaug_3.pt"
alexnet.load_state_dict(torch.load(model_name, map_location=device))
alexnet.eval()

logging.info(f"Starting Testing on Real data with model {model_name}")
accuracy = test_model(model=alexnet, validation_loader=test_loader)
print("Testing Complete.")