import clip
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Load the dataset
dataset= pd.read_pickle('ScriptDataset/TestDataset/BSTD/BSTD_test_HEP.pkl')
# dataset2 = pd.read_pickle('ScriptDataset/TrainDataset/BSTD/BSTD_train_HE.pkl')
# dataset = pd.concat([dataset1, dataset2], ignore_index=True)

# Load the fine-tuned model
model_path = 'models/clip/clip_finetuned_HEP_real.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ImageData class
class ImageData(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data.iloc[idx]
            image = item['image']
            subcategory = item['label']

            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    raise ValueError(f"Empty image data at index {idx}")
                if image.ndim == 3 and image.shape[2] == 4:
                    # RGBA image, convert to RGB
                    image = Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB')
                elif image.ndim == 3 and image.shape[2] == 3:
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                else:
                    raise ValueError(f"Unexpected image shape at index {idx}: {image.shape}")
            elif not isinstance(image, Image.Image):
                raise TypeError(f"Unexpected image type at index {idx}: {type(image)}")

            label = subcategories.index(subcategory)
            return self.transform(image), label
        except Exception as e:
            error_msg = f"Error accessing item at index {idx}: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            # Return a placeholder image and label
            placeholder_image = Image.new('RGB', (224, 224), color='gray')
            return self.transform(placeholder_image), -1

# Define subcategories
subcategories = list(dataset['label'].unique())

# Create DataLoader
val_loader = DataLoader(ImageData(dataset), batch_size=32, shuffle=False)
print("Created Dataloader")

# Load the CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define the model architecture
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

# Load the fine-tuned model
num_classes = len(subcategories)
model_ft = CLIPFineTuner(clip_model, num_classes)
model_ft.load_state_dict(torch.load(model_path))
model_ft = model_ft.to(device)
model_ft.eval()

print("Starting Testing")

# Testing loop
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Testing"):
        try:
            images, labels = images.to(device), labels.to(device)

            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        except Exception as e:
            print(f"Error in testing loop: {str(e)}")
            continue

# Calculate accuracy
total = len(all_labels)
correct = sum(np.array(all_labels) == np.array(all_predictions))
accuracy = 100 * correct / total

print(f'Accuracy: {accuracy:.2f}%')

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, labels=range(num_classes))

# Display precision, recall, and F1 for each class
for i, subcategory in enumerate(subcategories):
    print(f"Class: {subcategory}")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")

# Calculate and display macro-average precision, recall, and F1 score
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print("\nMacro-Average Metrics:")
print(f"  Precision: {macro_precision:.4f}")
print(f"  Recall: {macro_recall:.4f}")
print(f"  F1-Score: {macro_f1:.4f}")