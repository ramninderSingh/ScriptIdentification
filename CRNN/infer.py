from torchvision import transforms
from PIL import Image
from model import CRNN
import os
import torch
from config import infer_config as config

def load_model(model_path, num_classes):

    crnn = CRNN(3, 32, 64, num_classes, map_to_seq_hidden=32, rnn_hidden=256, leaky_relu=False)
    crnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=True))
    crnn.eval()
    return crnn

# model_path=input("Enter the model path: ")
model_path=config['model_path']


# print("Numbers assigned are as follows")
# print("1. Hindi,English")
# print("2. Hindi,English,Gujarati")
# print("3. Hindi,English,Punjabi")


# num=config['num']


# if num==1:
#     class_map={0:'Hindi',1:'English'}
# elif num == 2:
#     class_map={0:'Hindi',1:'English',2:'Gujarati'}
# elif num == 3:
#     class_map={0:'Hindi',1:'English',2:'Punjabi'}



transform = transforms.Compose([
    transforms.Resize((32, 64)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# crnn = load_model(model_path, len(class_map))
crnn = load_model(model_path, 3)


image=Image.open(config['img_path'])
image=transform(image).unsqueeze(0)

with torch.no_grad():
    logits_seq = crnn(image)  
    logits = torch.mean(logits_seq, dim=0)  
    prediction = torch.argmax(logits, dim=1).item()

# predicted_label = class_map[prediction]
if prediction==0:
    predicted_label="hindi"
elif prediction==1:
    predicted_label="english"
else:
     predicted_label="other language"



print(f"predicted_label: {predicted_label}, filename: {config['img_path']} ")

    


