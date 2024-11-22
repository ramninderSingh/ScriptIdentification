import torch
import clip
from PIL import Image
from io import BytesIO
import os
import requests
import time
import glob
import argparse
from tqdm import tqdm
import csv

# Model information dictionary containing model paths and language subcategories
model_info = {
    "hindi": {
        "path": "models/clip_finetuned_hindienglish_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglish_real.pth",
        "subcategories": ["hindi", "english"]
    },
    "assamese": {
        "path": "models/clip_finetuned_hindienglishassamese_real.pth",
        "url": "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishassamese_real.pth",
        "subcategories": ["hindi", "english", "assamese"]
    },
    "bengali": {
        "path": "models/clip_finetuned_hindienglishbengali_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishbengali_real.pth",
        "subcategories": ["hindi", "english", "bengali"]
    },
    "gujarati": {
        "path": "models/clip_finetuned_hindienglishgujarati_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishgujarati_real.pth",
        "subcategories": ["hindi", "english", "gujarati"]
    },
    "kannada": {
        "path": "models/clip_finetuned_hindienglishkannada_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishkannada_real.pth",
        "subcategories": ["hindi", "english", "kannada"]
    },
    "malayalam": {
        "path": "models/clip_finetuned_hindienglishmalayalam_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmalayalam_real.pth",
        "subcategories": ["hindi", "english", "malayalam"]
    },
    "marathi": {
        "path": "models/clip_finetuned_hindienglishmarathi_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmarathi_real.pth",
        "subcategories": ["hindi", "english", "marathi"]
    },
    "meitei": {
        "path": "models/clip_finetuned_hindienglishmeitei_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishmeitei_real.pth",
        "subcategories": ["hindi", "english", "meitei"]
    },
    "odia": {
        "path": "models/clip_finetuned_hindienglishodia_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishodia_real.pth",
        "subcategories": ["hindi", "english", "odia"]
    },
    "punjabi": {
        "path": "models/clip_finetuned_hindienglishpunjabi_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishpunjabi_real.pth",
        "subcategories": ["hindi", "english", "punjabi"]
    },
    "tamil": {
        "path": "models/clip_finetuned_hindienglishtamil_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishtamil_real.pth",
        "subcategories": ["hindi", "english", "tamil"]
    },
    "telugu": {
        "path": "models/clip_finetuned_hindienglishtelugu_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishtelugu_real.pth",
        "subcategories": ["hindi", "english", "telugu"]
    },
    "urdu": {
        "path": "models/clip_finetuned_hindienglishurdu_real.pth",
        "url" : "https://github.com/anikde/STscriptdetect/releases/download/V1/clip_finetuned_hindienglishurdu_real.pth",
        "subcategories": ["hindi", "english", "urdu"]
    }
}


# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class CLIPFineTuner(torch.nn.Module):
    """
    Fine-tuning class for the CLIP model to adapt to specific tasks.
    
    Attributes:
        model (torch.nn.Module): The CLIP model to be fine-tuned.
        classifier (torch.nn.Linear): A linear classifier to map features to the desired number of classes.
    """
    def __init__(self, model, num_classes):
        """
        Initializes the fine-tuner with the CLIP model and classifier.

        Args:
            model (torch.nn.Module): The base CLIP model.
            num_classes (int): The number of target classes for classification.
        """
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for image classification.

        Args:
            x (torch.Tensor): Preprocessed input tensor for an image.

        Returns:
            torch.Tensor: Logits for each class.
        """
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Extract image features from CLIP model
        return self.classifier(features)  # Return class logits

# Ensure model file exists; download directly if not
def ensure_model(model_name):
    model_path = model_info[model_name]["path"]
    url = model_info[model_name]["url"]

    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))  # Get total file size
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f, tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Downloaded model for {model_name}.")
    
    return model_path

# Prediction function to verify and load the model
def predict(image_path, model_name):
    """
    Predicts the class of an input image using a fine-tuned CLIP model.

    Args:
        image_path (str): Path to the input image file.
        model_name (str): Name of the model (e.g., hineng, hinengpun, hinengguj) as specified in `model_info`.

    Returns:
        dict: Contains either `predicted_class` if successful or `error` if an exception occurs.

    Example usage:
        result = predict("sample_image.jpg", "hinengguj")
        print(result)  # Output might be {'predicted_class': 'hindi'}
    """
    try:
        # Validate model name and retrieve associated subcategories
        if model_name not in model_info:
            return {"error": "Invalid model name"}

        # Ensure the model file is downloaded and accessible
        model_path = ensure_model(model_name)


        subcategories = model_info[model_name]["subcategories"]
        num_classes = len(subcategories)

        # Load the fine-tuned model with the specified number of classes
        model_ft = CLIPFineTuner(clip_model, num_classes)
        model_ft.load_state_dict(torch.load(model_path, map_location=device))
        model_ft = model_ft.to(device)
        model_ft.eval()

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Run the model and get the prediction
        outputs = model_ft(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = subcategories[predicted_idx.item()]

        return {"predicted_class": predicted_class}

    except Exception as e:
        return {"error": str(e)}

def predict_batch(image_dir, model_name, time_show, output_csv="predictions.csv"):
    """
    Processes all images in a directory to predict the class for each image using a specified fine-tuned CLIP model.
    
    Args:
        image_dir (str): Path to the directory containing image files to process.
        model_name (str): Name of the model (e.g., hineng, hinengpun, hinengguj) as specified in `model_info`.
        output_csv (str): Path to save the output CSV file. Defaults to 'predictions.csv'.
    
    Returns:
        str: Path to the CSV file containing predictions.
    """
    results = []
    
    # Validate model name and retrieve associated subcategories
    if model_name not in model_info:
        raise ValueError("Invalid model name.")
    
    # Ensure the model file is downloaded and accessible
    model_path = ensure_model(model_name)
    subcategories = model_info[model_name]["subcategories"]
    num_classes = len(subcategories)
    
    # Load the fine-tuned model with the specified number of classes
    model_ft = CLIPFineTuner(clip_model, num_classes)
    model_ft.load_state_dict(torch.load(model_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()
    
    # Get all image paths from the directory
    image_paths = glob.glob(os.path.join(image_dir, "*.*"))
    
    # Start timing the prediction process
    start_time = time.time()
    
    # Iterate over each image path in the batch
    for image_path in tqdm(image_paths):
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Run the model and get the prediction
            outputs = model_ft(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = subcategories[predicted_idx.item()]
            
            # Append the result to the list
            results.append({"Filepath": image_path, "Language": predicted_class})
        
        except Exception as e:
            # If an error occurs, log it in the results with the filename
            results.append({"Filepath": image_path, "Language": f"error: {str(e)}"})
    
    # Calculate and print the time taken to process the batch
    elapsed_time = time.time() - start_time
    if time_show:
        print(f"Time taken to process {len(image_paths)} images: {elapsed_time:.2f} seconds")
    
    # Write results to a CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filepath", "Language"])
        writer.writeheader()
        writer.writerows(results)
    
    return output_csv

if __name__ == "__main__":
    # Argument parser for command line usage
    parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--image_dir", type=str, help="Path to the input image directory")
    parser.add_argument("--model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")
    parser.add_argument("--batch", action="store_true", help="Process images in batch mode if specified")
    parser.add_argument("--time",type=bool, nargs="?", const=True, default=False, help="Prints the time required to process a batch of images")

    args = parser.parse_args()

    # Choose function based on the batch parameter
    if args.batch:
        if not args.image_dir:
            print("Error: image_dir is required when batch is set to True.")
        else:
            result = predict_batch(args.image_dir, args.model_name, args.time)
            print(result)
    else:
        if not args.image_path:
            print("Error: image_path is required when batch is not set.")
        else:
            result = predict(args.image_path, args.model_name)
            print(result)