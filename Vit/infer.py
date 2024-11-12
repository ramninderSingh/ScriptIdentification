from transformers import AutoImageProcessor,ViTForImageClassification,pipeline
from PIL import Image
from datasets import DatasetDict,Dataset,ClassLabel
import torchvision.transforms as transforms
import numpy as np
import csv
from config import infer_config as config
import os

trained_model=config['model_path']
pretrained_vit_model = config['pretrained_vit_model']
processor = AutoImageProcessor.from_pretrained(pretrained_vit_model,use_fast=True)
vit = ViTForImageClassification.from_pretrained(trained_model)
model = pipeline('image-classification', model=vit, feature_extractor=processor,device=0)



image_path=config['img_path']

folder_path = config['folder_path']


if image_path.endswith((".png", ".jpg", ".jpeg")):  
    image = Image.open(image_path)
    output = model(image)
    predicted_label = max(output, key=lambda x: x['score'])['label']
    
    print(f"image_path: {image_path}, original_label: 'telugu', predicted_label: {predicted_label}\n")
    
else:
    print("No image provided")
    print("Checking folder")
    output_csv_path = config['csv_path']
    base_prefix = "/DATA1/ocrteam/"

    # with open(output_file_path, 'a') as file:
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filepath', 'predicted_class'])
        for filename in os.listdir(folder_path):
            
            if filename.endswith((".png", ".jpg", ".jpeg")):  
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path)
                
                if img_path.startswith(base_prefix):
                    img_path = img_path[len(base_prefix):]

                output = model(image)
                predicted_label = max(output, key=lambda x: x['score'])['label'].capitalize()
                
                csv_writer.writerow([img_path, predicted_label])
                print(f"Processed {filename} -> Predicted class: {predicted_label}")

    print(f"Predictions saved to {output_csv_path}")














