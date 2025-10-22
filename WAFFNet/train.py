import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from utils.dataset import RecognitionDataset,ImageData, create_dataloaders
from utils.transforms import get_train_transforms,get_val_transforms
from models.waffnet import WAFFNetPP
from utils.losses import FocalLoss
import torch.optim as optim
from torch.cuda.amp import GradScaler,autocast
import copy
from torchsummary import summary



base_path = '/content/augmented_dataset_5000/augmented_dataset_5000'
folders = ['hindi', 'english', 'odia','assamese', 'bengali','telugu','tamil','gujarati','marathi','kannada','malayalam','punjabi']

data = []
valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for root, _, files in os.walk(folder_path):
        for filename in tqdm(files, desc=f"Collecting {folder}"):
            if filename.lower().endswith(valid_exts):
                image_path = os.path.join(root, filename)
                label = folder  # folder name as label
                data.append({'image_path': image_path, 'label': label})

df = pd.DataFrame(data)

print(f"Dataset index built with {len(df)} entries.")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# Save lightweight dataset index
os.makedirs("all_train", exist_ok=True)
file_name = "all_train/recognition_index.pkl"
df.to_pickle(file_name)
print(f"Index saved as '{file_name}'")

# Define PyTorch Dataset with augmentations
STANDARD_SIZE = (224, 224)

# Load dataset index
dataset = pd.read_pickle("all_train/recognition_index.pkl")
print(f"Dataset shape: {dataset.shape}")

subcategories = sorted(dataset['label'].unique().tolist())  # class names

# Dataset class + transforms
train_transforms = get_train_transforms()
val_transforms = get_val_transforms()
# Dataloaders
train_loader,val_loader=create_dataloaders(dataset, train_transforms,val_transforms,subcategories,batch_size=32)
print("Created Dataloaders")


# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WAFFNetPP(num_classes=len(subcategories)).to(device)

criterion = FocalLoss(gamma=2.0)  # focal loss

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-6
)

scaler = GradScaler()  # for mixed precision
summary(model, input_size=(1, 3, 32, 32))

# Training settings
max_epochs = 100
patience = 10
best_val_loss = float('inf')
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())


# Training Loop
for epoch in range(max_epochs):
    model.train()
    train_loss, train_correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")

    for batch in pbar:
        images, labels = batch[:2]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():  # mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # stats
        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100*train_correct/total:.2f}%"
        })

    avg_train_loss = train_loss / total
    avg_train_acc = train_correct / total


    # validation
    model.eval()
    val_loss, val_correct, total_val = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
            images, labels = batch[:2]
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / total_val
    avg_val_acc = val_correct / total_val


    print(f"\nEpoch {epoch+1}/{max_epochs}: "
          f"Train Loss {avg_train_loss:.4f}, Train Acc {avg_train_acc*100:.2f}% | "
          f"Val Loss {avg_val_loss:.4f}, Val Acc {avg_val_acc*100:.2f}%\n")

    scheduler.step()
    #  Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f" Early stopping triggered after {epoch+1} epochs.")
            break

# Save Best Model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "waffnetpp_best_cbam.pth")
print(" Training finished & best model saved.")



