"""
test.py — WAFFNet++ Evaluation Script (Config-Driven)
-----------------------------------------------------
Evaluates WAFFNet++ on a labeled test dataset directory.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict

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


def build_test_dataframe(test_dir: str, class_names: List[str]) -> pd.DataFrame:
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    data = []

    for cls in class_names:
        folder_path = os.path.join(test_dir, cls)
        if not os.path.isdir(folder_path):
            print(f"[Warning] Missing class directory: {folder_path}")
            continue

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(valid_exts):
                    data.append({
                        "image_path": os.path.join(root, filename),
                        "label": cls
                    })

    df = pd.DataFrame(data)
    print(f"\n[INFO] Found {len(df)} test images.")
    print(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def evaluate_model(
    test_dir: str = TEST_DATA_PATH,
    weights_path: str = MODEL_SAVE_PATH,
    weights_id: str = MODEL_DRIVE_ID
) -> None:

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # Step 1 — Dataset and DataLoader
    df = build_test_dataframe(test_dir, LANGUAGES)
    class_names = sorted(df["label"].unique())
    test_transform = get_val_transforms()
    test_loader = create_test_loader(df, test_transform, class_names, batch_size=32)

    # Step 2 — Load trained model
    model = load_model(weights_path, weights_id, device)
    model.eval()

    # Step 3 — Inference loop
    all_labels, all_preds = [], []
    results_dict: Dict[str, Dict[str, str]] = {}

    print("\n[INFO] Starting evaluation...")

    with torch.no_grad():
        for images, labels, filenames in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for fname, pred, label in zip(filenames, preds.cpu().numpy(), labels.cpu().numpy()):
                results_dict[fname] = {
                    "True_Language": class_names[label],
                    "Predicted_Language": class_names[pred],
                }

    # Step 4 — Save predictions
    results_df = pd.DataFrame(
        [{"Filename": k, **v} for k, v in results_dict.items()]
    )
    csv_path = os.path.join(RESULTS_DIR, "predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Predictions saved to: {csv_path}")

    # Step 5 — Compute metrics and confusion matrix
    true_labels, pred_labels = np.array(all_labels), np.array(all_preds)
    valid_mask = true_labels != -1
    true_labels, pred_labels = true_labels[valid_mask], pred_labels[valid_mask]

    compute_metrics(true_labels, pred_labels, class_names)
    plot_confusion_matrix(
        true_labels,
        pred_labels,
        class_names,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )

    # Step 6 — Optional: Visualization
    # test_dataset = TestDataset(df, test_transform, class_names)
    # visualize_predictions_per_class(model, test_dataset, class_names, device)

    print("\n[SUCCESS] Evaluation completed successfully.")


if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {str(e)}")
