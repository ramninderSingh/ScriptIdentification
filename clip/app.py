from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
import clip
from io import BytesIO
import os

app = FastAPI()

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
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class CLIPFineTuner(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = torch.nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float() 
        return self.classifier(features)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        if model_name not in model_info:
            return {"error": "Invalid model name"}

        model_path = model_info[model_name]["path"]
        subcategories = model_info[model_name]["subcategories"]
        num_classes = len(subcategories)

        model_ft = CLIPFineTuner(clip_model, num_classes)
        model_ft.load_state_dict(torch.load(model_path, map_location=device))
        model_ft = model_ft.to(device)
        model_ft.eval()

        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        outputs = model_ft(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = subcategories[predicted_idx.item()]

        return {"predicted_class": predicted_class}

    except Exception as e:
        return {"error": str(e)}

