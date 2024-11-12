import torch
import clip
from PIL import Image
import pandas as pd
import os

lang = "Marathi"
# Define model info dictionary
model_info = {
    "temp": {
        "path": "all_models/clip_finetuned_hindienglishmarathi_real.pth",
        "subcategories": ["Hindi", "English",lang]
    },
    "hineng": {
        "path": "models/clip/clip_finetuned_HE_real.pth",
        "subcategories": ["Hindi", "English"]
    },
    "hinengguj": {
        "path": "models/clip/clip_finetuned_HEP_real.pth",
        "subcategories": ["Hindi", "English", "Gujarati"]
    },
    "all": {
        "path": "all_models/clip_finetuned_hindienglishassamesebengaligujaratikannadamalayalammarathimeiteiodiapunjabitamiltelugu_real.pth",
        "subcategories": [
            'hindi', 'english', 'bengali', 'gujarati', 'kannada', 'odia', 
            'punjabi', 'tamil', 'telugu', 'assamese', 'malayalam', 'meitei', 'urdu'
        ]
    }
}

# Set device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define fine-tuner class
class CLIPFineTuner(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Extract features
        return self.classifier(features)  # Return class logits

# Function to predict the class of a single image
def predict_image(image_path, model_ft, subcategories):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Run the model and get the prediction
        outputs = model_ft(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = subcategories[predicted_idx.item()]

        return predicted_class
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle folder or single image prediction
def predict(input_path, model_name):
    # Get model details
    if model_name not in model_info:
        return {"error": "Invalid model name"}

    subcategories = model_info[model_name]["subcategories"]
    model_path = model_info[model_name]["path"]
    num_classes = len(subcategories)

    # Load the fine-tuned model
    model_ft = CLIPFineTuner(clip_model, num_classes)
    model_ft.load_state_dict(torch.load(model_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()
    path = "test_csv"
    # If input path is a directory, process all images in it
    if os.path.isdir(input_path):
        predictions = []
        
        for filename in os.listdir(input_path):
            image_path = os.path.join(input_path, filename)
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue  # Skip non-image files

            predicted_class = predict_image(image_path, model_ft, subcategories)
            predictions.append({"image_path": image_path, "predicted_class": predicted_class})

        # Save predictions to CSV
        df = pd.DataFrame(predictions)
        output_csv_path = os.path.join(path, f"predictions_{lang}.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")

    # If input path is a single image, predict for that image
    elif os.path.isfile(input_path):
        predicted_class = predict_image(input_path, model_ft, subcategories)
        print(f"Image: {input_path}, Predicted Class: {predicted_class}")
    else:
        print("Error: The specified path is neither a file nor a directory.")

# If this file is run directly, accept inputs
if __name__ == "__main__":
    import argparse

    # Argument parser to take inputs from the command line
    parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
    parser.add_argument("input_path", type=str, help="Path to the input image or folder")
    parser.add_argument("model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")

    args = parser.parse_args()

    # Call the predict function with inputs
    predict(args.input_path, args.model_name)
