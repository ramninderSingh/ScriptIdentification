"""
config.py — Central configuration for WAFFNet++ training
Author: Ramninder Singh
Project: ScriptIdentification (WAFFNet++)
-------------------------------------------
This file holds all training parameters and dataset paths.
Modify these values to retrain or fine-tune the model.
"""

import torch
from datetime import datetime

# =====================
#  PATHS
# =====================

# Base dataset path (change this to your local or cloud dataset directory)
DATASET_BASE_PATH = r"C:\Users\HP\Desktop\BTP\Git\augmented_dataset_5000"

# Location where the dataset index will be saved
TRAIN_INDEX_PATH = "all_train/recognition_index.pkl"

# Model checkpoint save path
MODEL_SAVE_PATH = "weights/waffnetpp_best_cbam.pth"

# Logging directory
LOG_DIR = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

# =====================
#  TRAINING PARAMETERS
# =====================

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 3
FOCAL_GAMMA = 2.0
T_MAX = 10
ETA_MIN = 1e-6
GRAD_CLIP = 5.0

# =====================
#  MODEL / DEVICE
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12  # can be inferred dynamically
PRINT_MODEL_SUMMARY = True

# =====================
#  AUGMENTATION SETTINGS
# =====================

RANDOM_ROTATION = 5
COLOR_JITTER = (0.3, 0.3, 0.2, 0.05)
RANDOM_AFFINE = (0.1, 0.1)
PERSPECTIVE_DISTORTION = 0.3
