"""
evaluations.py — Evaluation utilities for WAFFNet++
---------------------------------------------------
Contains reusable functions for model evaluation and reporting.
"""

import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ===============================================================
#  Core Metrics
# ===============================================================

def compute_metrics(true_labels, pred_labels, class_names):
    """Compute and print overall + per-class accuracy."""
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\n Overall Accuracy: {accuracy * 100:.2f}%\n")

    print("Class-wise Accuracy:")
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))

    for t, p in zip(true_labels, pred_labels):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    rows = []
    for i, label in enumerate(class_names):
        acc = (class_correct[i] / class_total[i]) * 100 if class_total[i] > 0 else 0
        rows.append({"Class": label, "Accuracy (%)": round(acc, 2),
                     "Correct": int(class_correct[i]), "Total": int(class_total[i])})
        print(f"  {label:<12}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")

    # Save class-wise accuracy CSV
    df_acc = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df_acc.to_csv("results/classwise_accuracy.csv", index=False)
    print("Class-wise accuracy saved to: results/classwise_accuracy.csv")

    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=3, output_dict=False)
    print("\nDetailed Classification Report:\n")
    print(classification_report(true_labels, pred_labels, target_names=class_names, digits=3))

    return accuracy

# ===============================================================
#  Confusion Matrix
# ===============================================================

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path=None):
    """Display and optionally save confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Confusion matrix saved at: {save_path}")

    plt.show()


# ===============================================================
#  Save Predictions (CSV)
# ===============================================================

def save_predictions_csv(results_dict, output_path="results/predictions.csv"):
    """Save predictions as CSV: filename, predicted_label."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(list(results_dict.items()), columns=["Filename", "Predicted_Label"])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


# ===============================================================
#  Visualization
# ===============================================================

def visualize_predictions_per_class(model, dataset, subcategories, device, examples_per_class=5):
    """Show sample predictions for each language."""
    import random
    print("\n Sample predictions per language:\n")
    model.eval()

    for label_idx, lang in enumerate(subcategories):
        indices = [i for i, row in enumerate(dataset.data['label']) if row == lang]
        if not indices:
            continue
        selected_indices = random.sample(indices, min(examples_per_class, len(indices)))

        fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 3))
        if len(selected_indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, selected_indices):
            img, label, fname = dataset[idx]
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(device))
                pred_label = torch.argmax(pred, 1).item()

            img_vis = img.permute(1, 2, 0).cpu().numpy()
            img_vis = np.clip(img_vis * np.array((0.2686, 0.2613, 0.2758)) +
                              np.array((0.4815, 0.4578, 0.4082)), 0, 1)

            color = "green" if subcategories[pred_label] == subcategories[label] else "red"
            ax.imshow(img_vis)
            ax.axis("off")
            ax.set_title(f"Pred: {subcategories[pred_label]}\nTrue: {subcategories[label]}", color=color, fontsize=9)

        plt.suptitle(f"Language: {lang}", fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()
