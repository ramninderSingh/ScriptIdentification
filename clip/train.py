import pandas as pd
import clip
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import numpy as np

# Load the dataset
dataset = pd.read_pickle('ScriptDataset/TrainDataset/BSTD/')
print(f"Dataset shape: {dataset.shape}")


subcategories = list(dataset['label'].unique())
model, preprocess = clip.load("ViT-B/32", jit=False)

# Split the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Created Dataset")

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
            if isinstance(self.data, torch.utils.data.Subset):
                item = self.data.dataset.iloc[self.data.indices[idx]]
            else:
                item = self.data.iloc[idx]
            image = item['image']
            subcategory = item['label']
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    raise ValueError(f"Empty image data at index {idx}")
                if image.ndim == 3 and image.shape[2] == 4:
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
            placeholder_image = Image.new('RGB', (224, 224), color='gray')
            return self.transform(placeholder_image), -1

subcategories = list(dataset['label'].unique())

train_loader = DataLoader(ImageData(train_dataset), batch_size=32, shuffle=True)
val_loader = DataLoader(ImageData(val_dataset), batch_size=32, shuffle=False)
print("Created Dataloader")

class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.classifier(features)

num_classes = len(subcategories)
model_ft = CLIPFineTuner(model, num_classes)
# model_ft.load_state_dict(torch.load('clip_finetuned_HE_syn.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)



print("Starting Training")
num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    model_ft.train() 
    running_loss = 0.0 
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000") 

    for images, labels in pbar:
        try:
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()  
            outputs = model_ft(images) 
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()
            running_loss += loss.item()  
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")  
        except Exception as e:
            print(f"Error in training loop: {str(e)}")
            continue

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  
    model_ft.eval()  
    correct = 0 
    total = 0 

    with torch.no_grad():  
        for images, labels in val_loader:
            try:
                images, labels = images.to(device), labels.to(device)  
                outputs = model_ft(images) 
                _, predicted = torch.max(outputs.data, 1) 
                total += labels.size(0) 
                correct += (predicted == labels).sum().item() 
            except Exception as e:
                print(f"Error in validation loop: {str(e)}")
                continue

    print(f'Validation Accuracy: {100 * correct / total}%')  

torch.save(model_ft.state_dict(), 'clip_finetuned_HE_synreal')