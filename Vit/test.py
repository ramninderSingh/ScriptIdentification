from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_score, recall_score, f1_score
from EDA import ImageDataset4
from config import test_config as config
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor,ViTForImageClassification,pipeline
from PIL import Image
import torch
from datasets import DatasetDict,Dataset,ClassLabel
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from torch import no_grad




# Loading Dataset and Labels
data_folders = {
    'hindi': config['hindi_path_real'],
    'english': config['english_path_real'],
    'gujarati': config['gujarati_path_real'],
    'punjabi':config['punjabi_path_real'],
    'assamese':config['assamese_path_real'],
    'bengali':config['bengali_path_real'],
    'kannada':config['kannada_path_real'],
    'malayalam':config['malayalam_path_real'],
    'marathi':config['marathi_path_real'],
    'odia':config['odia_path_real'],
    'tamil':config['tamil_path_real'],
    'telugu':config['telugu_path_real']
}


print(data_folders.keys())

# def getPrediction(image):
#     image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     result = model(image, top_k=1)
#     return result[0]['label'] 


# y_true, y_pred = [], []

# for label, folder in data_folders.items():
#     for image_file in os.listdir(folder):
#         image_path = os.path.join(folder, image_file)
#         if os.path.isfile(image_path) and image_file.endswith(('.png', '.jpg', '.jpeg')):
#             image = cv2.imread(image_path)
#             predicted_label = getPrediction(image)
            
#             y_true.append(label)
#             y_pred.append(predicted_label) 

dataset = ImageDataset4(data_folders,max_images=config['max_images'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


save_dir=config['reload_model']
model_name_or_path = config['pretrained_vit_model']
processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=True)
vit = ViTForImageClassification.from_pretrained(save_dir)
model = pipeline('image-classification', model=vit, feature_extractor=processor, device=0)


y_true, y_pred = [], []

# Iterate over the dataloader
for images, labels in dataloader:
    with no_grad():
        # Use the processor to prepare batch
        pil_images = [transforms.ToPILImage()(img) for img in images] 
        outputs = model(pil_images)




        predicted_labels = []


        for output in outputs:
            # Get the label with the highest score
            predicted_label = max(output, key=lambda x: x['score'])['label']
            predicted_labels.append(predicted_label)  # Append the predicted label

        # Convert predicted labels to indices
        predicted_labels_indices = [list(data_folders.keys()).index(label) for label in predicted_labels]

        # Append true and predicted labels
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels_indices)

labels=[]
for i in range(config['classes']):
    labels.append(i)



# Convert lists to numpy arrays for evaluation
y_true = np.array(y_true)
y_pred = np.array(y_pred)


plt.figure(figsize=(20, 15))
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(data_folders.keys()))
disp.plot(xticks_rotation=45)

plt.tight_layout()

output_path = f"{save_dir}/confusion_matrix.png"
plt.savefig(output_path, format='png')
plt.close()

# recall = recall_score(y_true, y_pred, average=None, labels=list(data_folders.keys()))
# for label, score in zip(data_folders.keys(), recall):
#     print(f"Recall for {label}: {score:.2f}")

accuracy = np.sum(y_true == y_pred) / len(y_true)  # Correct predictions divided by total predictions
print(np.sum(y_true==y_pred))
print(len(y_true))
print(f"Accuracy: {accuracy:.4f}")  # Print accuracy


precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')



print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')


total_images = {}
correct_pred_dict={}
for i, label in enumerate(data_folders.keys()):
    # Correct predictions for this class
    correct_preds = np.sum((y_true == i) & (y_pred == i))
    # Total instances for this class
    total_preds = np.sum(y_true == i)
    correct_pred_dict[label]=correct_preds
    total_images[label]=total_preds
    # # Avoid division by zero for classes with no samples
    # class_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    # class_accuracies[label] = class_accuracy

# Print per-class accuracy
for label, images in total_images.items():
    correct_preds=correct_pred_dict[label]
    print(f"Correct Predictions for{label}: {correct_preds}/{images}")
    acc=correct_preds/images
    print(f"Accuracy for {label} : {acc:.2f}")