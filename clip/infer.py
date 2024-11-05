import torch
import clip
from PIL import Image
from io import BytesIO
import os

# Define model info dictionary
model_info = {
    "hinengpun": {
        "path": "models/clip/clip_finetuned_HEP_real.pth",
        "subcategories": ["Hindi", "English", "Punjabi"]
    },
    "hineng": {
        "path": "models/clip/clip_finetuned_HE_real.pth",
        "subcategories": ["Hindi", "English"]
    },
    "hinengguj": {
        "path": "models/clip/clip_finetuned_HEP_real.pth",
        "subcategories": ["Hindi", "English", "Gujarati"]
    },
    "all" : {
        "path": "all_models/clip_finetuned_hindienglishassamesebengaligujaratikannadamalayalammarathimeiteiodiapunjabitamiltelugu_real.pth",
        "subcategories": ['hindi','english','bengali','gujarati','kannada','odia','punjabi','tamil','telugu','assamese','malayalam','meitei','urdu']
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

# Function to predict the class of an image
def predict(image_path, model_name):
    try:
        # Get the subcategories based on the model name
        if model_name not in model_info:
            return {"error": "Invalid model name"}

        subcategories = model_info[model_name]["subcategories"]
        model_path = model_info[model_name]["path"]
        num_classes = len(subcategories)

        # Load the fine-tuned model
        model_ft = CLIPFineTuner(clip_model, num_classes)
        try:
            model_ft.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
        except Exception as e:
            return {"error": f"Failed to load model from path {model_path}: {str(e)}"}
        
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

# If this file is run directly, accept inputs
if __name__ == "__main__":
    import argparse

    # Argument parser to take inputs from the command line
    parser = argparse.ArgumentParser(description="Image classification using CLIP fine-tuned model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("model_name", type=str, choices=model_info.keys(), help="Name of the model (e.g., hineng, hinengpun, hinengguj)")

    args = parser.parse_args()

    # Call the predict function with inputs
    result = predict(args.image_path, args.model_name)
    print(result)
