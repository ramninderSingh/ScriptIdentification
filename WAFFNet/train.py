"""
train.py — Training script for WAFFNet++
----------------------------------------
Trains the WAFFNet++ model for script identification.
Uses Focal Loss, cosine annealing LR scheduler, and mixed precision.


"""
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import copy

# Local imports
from utils.dataset import create_dataloaders
from utils.transforms import get_train_transforms, get_val_transforms
from models.waffnet import WAFFNetPP
from utils.losses import FocalLoss
from config import (
    TRAIN_DATA_PATH,
    TRAIN_INDEX_PATH,
    MODEL_SAVE_PATH,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    PATIENCE,
    FOCAL_GAMMA,
    T_MAX,
    ETA_MIN,
    GRAD_CLIP,
    PRINT_MODEL_SUMMARY
)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


#  Dataset Preparation
def build_dataset_index(base_path, save_path):
    """Creates and saves a dataset index as a DataFrame."""
    from PIL import Image
    from tqdm import tqdm

    folders = [
        'hindi', 'english', 'odia', 'assamese', 'bengali',
        'telugu', 'tamil', 'gujarati', 'marathi', 'kannada',
        'malayalam', 'punjabi'
    ]
    valid_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    data = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for root, _, files in os.walk(folder_path):
            for filename in tqdm(files, desc=f"Collecting {folder}"):
                if filename.lower().endswith(valid_exts):
                    image_path = os.path.join(root, filename)
                    label = folder
                    data.append({'image_path': image_path, 'label': label})

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)
    print(f"Dataset index built with {len(df)} samples.")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


#  Training Function
def train():
    # Ensure dataset index exists
    if not os.path.exists(TRAIN_INDEX_PATH):
        dataset = build_dataset_index(TRAIN_DATA_PATH, TRAIN_INDEX_PATH)
    else:
        dataset = pd.read_pickle(TRAIN_INDEX_PATH)
        print(f"Loaded dataset index: {dataset.shape[0]} samples.")

    subcategories = sorted(dataset['label'].unique().tolist())
    print(f"Detected classes: {subcategories}")

    # Transforms & Dataloaders
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    train_loader, val_loader = create_dataloaders(
        dataset, train_transforms, val_transforms, subcategories, batch_size=BATCH_SIZE
    )
    print("Created dataloaders successfully.\n")

    # Initialize model, loss, optimizer, scheduler
    model = WAFFNetPP(num_classes=len(subcategories)).to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
    scaler = GradScaler()

    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n Starting training...\n")
    for epoch in range(MAX_EPOCHS):
        # Training 
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            # Stats
            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / total:.2f}%"
            })

        avg_train_loss = train_loss / total
        avg_train_acc = train_correct / total


        # Validation 
        model.eval()
        val_loss, val_correct, total_val = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        avg_val_acc = val_correct / total_val

        print(f"\n Epoch {epoch+1}/{MAX_EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc*100:.2f}%\n")

        scheduler.step()

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f" Early stopping triggered after {epoch+1} epochs.")
                break

    # Save Best Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" Training finished. Best model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
