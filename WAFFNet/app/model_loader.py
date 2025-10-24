import torch
import os
from models.waffnet import WAFFNetPP
from config import MODEL_SAVE_PATH, MODEL_DRIVE_ID, DEVICE
from infer import load_model

# Load the model once at startup
def get_model():
    device = torch.device(DEVICE)
    model = load_model(MODEL_SAVE_PATH, MODEL_DRIVE_ID, device)
    return model, device
