from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, ViTForImageClassification, pipeline
import torch
from config import common_config as config

# Load the trained model and processor
pretrained_vit_model = config['pretrained_vit_model']
trained_model = 'saved_models_vit/15_epochs/best_model'

processor = AutoImageProcessor.from_pretrained(pretrained_vit_model, use_fast=True)
vit = ViTForImageClassification.from_pretrained(trained_model)
model = pipeline('image-classification', model=vit, feature_extractor=processor, device=0)

# Initialize the FastAPI app
app = FastAPI()

@app.post("/predict_lang/")
async def classify_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Classify the image using the model
    output = model(image)
    predicted_label = max(output, key=lambda x: x['score'])['label']

    # Return the result as a JSON response
    return {
        "image_path": file.filename,
        "predicted_label": predicted_label
    }
