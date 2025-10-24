import torch
from PIL import Image
from utils.transforms import get_val_transforms
from config import LANGUAGES

transform = get_val_transforms()

def predict(model, device, image: Image.Image):
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return LANGUAGES[pred.item()]
