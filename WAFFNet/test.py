"""
test.py — WAFFNet++ Evaluation Script (Config-Driven)
-----------------------------------------------------
Evaluates WAFFNet++ on a labeled test dataset directory.
Uses infer.py for model loading and utilities for evaluation.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.waffnet import WAFFNetPP
from utils.dataset import create_test_loader, TestDataset
from utils.transforms import get_val_transforms
from utils.evaluations import (
    compute_metrics,
    plot_confusion_matrix,
    save_predictions_csv,
    visualize_predictions_per_class,
)
from infer import load_model
from config import (
    TEST_DATA_PATH,
    MODEL_SAVE_PATH,
    MODEL_DRIVE_ID,
    DEVICE,
    IMAGE_SIZE,
    LANGUAGES,
    RESULTS_DIR,
)


# ============================================================
#  Build test dataframe
# ============================================================
def build_test_dataframe(test_dir, class_names):
    """Scans test directory and builds a DataFrame with image paths + labels."""
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    data = []

    for folder in class_names:
        folder_path = os.path.join(test_dir, folder)
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(valid_exts):
                    image_path = os.path.join(root, filename)
                    data.append({"image_path": image_path, "label": folder})

    df = pd.DataFrame(data)
    print(f"\n Found {len(df)} test images.")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


# ============================================================
#  Evaluation function
# ============================================================
def evaluate_model(test_dir=TEST_DATA_PATH, weights_path=MODEL_SAVE_PATH, weights_id=MODEL_DRIVE_ID):
    """Run WAFFNet++ evaluation using config-driven paths."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1 — Build dataset and dataloader
    df = build_test_dataframe(test_dir, LANGUAGES)
    subcategories = sorted(df["label"].unique())
    num_classes = len(subcategories)

    test_transform = get_val_transforms()
    test_loader = create_test_loader(df, test_transform, subcategories, batch_size=32)

    # Step 2 — Load model (from infer.py)
    device = torch.device(DEVICE)
    model = load_model(weights_path, weights_id, device)

    # Step 3 — Inference loop (with single-line tqdm)
    all_labels, all_preds = [], []
    results_dict = {}

    with torch.no_grad():
        for images, labels, filenames in tqdm(
            test_loader,
            desc="Evaluating",
            ascii=True,       # compatible progress bar characters
            ncols=100,        # fit nicely in most terminals
            leave=False,      # remove the bar after completion
            dynamic_ncols=True  # auto-adjust width dynamically
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for fname, pred, label in zip(filenames, preds.cpu().numpy(), labels.cpu().numpy()):
                results_dict[fname] = {
                    "True_Label": subcategories[label],
                    "Predicted_Label": subcategories[pred],
                }

    # Step 4 — Save predictions CSV
    results_df = pd.DataFrame(
        [{"Filename": k, **v} for k, v in results_dict.items()]
    )
    csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n Predictions saved to {csv_path}")

    # Step 5 — Metrics and confusion matrix
    true_labels = np.array(all_labels)
    pred_labels = np.array(all_preds)
    mask = true_labels != -1
    true_labels, pred_labels = true_labels[mask], pred_labels[mask]

    compute_metrics(true_labels, pred_labels, subcategories)
    plot_confusion_matrix(
        true_labels,
        pred_labels,
        subcategories,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )

    # Step 6 — Visualization
    test_dataset = TestDataset(df, test_transform, subcategories)
    visualize_predictions_per_class(model, test_dataset, subcategories, device)

    print("\n Evaluation completed successfully.")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    evaluate_model()
