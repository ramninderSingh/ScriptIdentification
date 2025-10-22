import os
import argparse
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from models.waffnet import WAFFNetPP
import pandas as pd
from datetime import datetime
from utils.transforms import get_val_transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LANGUAGES = [
    'assamese', 'bengali', 'english', 'gujarati', 'hindi',
    'kannada', 'malayalam', 'marathi', 'odia',
    'punjabi', 'tamil', 'telugu'
]

TRANSFORM = get_val_transforms()

def load_model(weights_path: str, file_id: str, device: str):
    """Load WAFFNet model, downloading weights if missing."""
    folder = os.path.dirname(weights_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Downloading model weights to {weights_path}...")
        gdown.download(id=file_id, output=weights_path, quiet=False)

    model = WAFFNetPP(num_classes=len(LANGUAGES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def predict_image(model, image_path, device):
    """Predict the script label for a single image."""
    img = Image.open(image_path).convert("RGB")
    img = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return LANGUAGES[pred.item()]


def predict_folder(model, folder_path, device):
    """Run inference on all images in a folder and save results as CSV."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    results = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

    for fname in tqdm(image_files, desc="Inferring"):
        img_path = os.path.join(folder_path, fname)
        label = predict_image(model, img_path, device)
        results.append({
            "Filepath": os.path.abspath(img_path),
            "Predicted Language": label
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create an output folder for CSVs
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("results", f"inference_results_{timestamp}.csv")

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"\n Results saved to: {csv_path}\n")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAFFNet++ Inference")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to an image or a folder containing images.")
    parser.add_argument("--weights_path", type=str, default="weights/waffnet_best_cbam.pth",
                        help="Path to model weights (.pth).")
    parser.add_argument("--weights_id", type=str, default="1McEiKujTfxCBfwhAsnTG1P2uJw8OaDbj",
                        help="Google Drive file ID for gdown fallback.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights_path, args.weights_id, device)

    if os.path.isdir(args.data_path):
        results = predict_folder(model, args.data_path, device)
        print("\nPredictions:")
        for fname, label in results.items():
            print(f"{fname:30s} → {label}")
    else:
        label = predict_image(model, args.data_path, device)
        print(f"\nPrediction for {args.data_path}: {label}")