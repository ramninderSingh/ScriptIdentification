from transformers import AutoImageProcessor,ViTForImageClassification,pipeline
from PIL import Image
from datasets import DatasetDict,Dataset,ClassLabel
import torchvision.transforms as transforms
import numpy as np


# trained_model='saved_models_vit/15_epochs/best_model'
trained_model='best_model'
pretrained_vit_model = 'google/vit-base-patch16-224-in21k'
processor = AutoImageProcessor.from_pretrained(pretrained_vit_model,use_fast=True)
vit = ViTForImageClassification.from_pretrained(trained_model)
model = pipeline('image-classification', model=vit, feature_extractor=processor,device=0)



image_path="image.png"

if image_path.endswith((".png", ".jpg", ".jpeg")):  
    image = Image.open(image_path)
    output = model(image)
    predicted_label = max(output, key=lambda x: x['score'])['label']
    
    print(f"image_path: {image_path}, original_label: 'telugu', predicted_label: {predicted_label}\n")
    
else:
    print("Provided filepath not of image")



