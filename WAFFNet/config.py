"""
config.py — Central configuration for WAFFNet++
Author: Ramninder Singh
Project: Script Identification
---------------------------------
Holds all paths, hyperparameters, and device settings.
"""
import os
import torch
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
#  PATHS
# =====================================================
STANDARD_SIZE = (224, 224)
# Dataset base paths
TRAIN_DATA_PATH = r"C:\Users\HP\Desktop\BTP\Git\augmented_dataset_5000"  #Replace with your training data path
TEST_DATA_PATH = r"C:\Users\HP\Desktop\BTP\recognition\recognition\test_478"  #Replace with your test data path

# Index files
TRAIN_INDEX_PATH = os.path.join(BASE_DIR, "all_train", "recognition_index.pkl")
TEST_INDEX_PATH = os.path.join(BASE_DIR, "all_train", "recognition_test_index.pkl")


# Model checkpoints
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "weights", "waffnet_best_cbam.pth")

# Google Drive backup ID (for auto-download via gdown)
MODEL_DRIVE_ID = "1McEiKujTfxCBfwhAsnTG1P2uJw8OaDbj"

# Output and logging
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


# =====================================================
#  TRAINING PARAMETERS
# =====================================================

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
FOCAL_GAMMA = 2.0
T_MAX = 10
ETA_MIN = 1e-6
GRAD_CLIP = 5.0

# =====================================================
#  MODEL / DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12  # can be inferred dynamically
PRINT_MODEL_SUMMARY = True

# =====================================================
#  CLASS LABELS
# =====================================================

LANGUAGES = [
    "assamese", "bengali", "english", "gujarati", "hindi",
    "kannada", "malayalam", "marathi", "odia",
    "punjabi", "tamil", "telugu",
]
