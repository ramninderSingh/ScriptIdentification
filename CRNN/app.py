from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import JSONResponse   
from torchvision import transforms
from PIL import Image
from model import CRNN
import os
import torch




app=FastAPI()


items=[]


@app.post("/items")
def create_item(item:str):
    items.append(item)
    return items

@app.post("/items/{item_id}")
def get_item(item_id:int)->str:
    if item_id<len(items):
        return items[item_id]
    
    else:
        return HTTPException(status_code=404,detail=f"Item {item_id} not found")


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the file locally or process it
        file_location = f"uploaded_images/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        return JSONResponse(content={"filename": file.filename, "message": "Image uploaded successfully"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


transform = transforms.Compose([
    transforms.Resize((32, 64)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define model path and label mappings
model_info = {
    "hinengpun": {"path": "savedModels/HEP/crnn_real_t2.pt", "classes": {0: "Hindi", 1: "English", 2: "Punjabi"}},
    "hineng": {"path": "savedModels/HE/crnn_real_t2.pt", "classes": {0: "Hindi", 1: "English"}},
    "hinengguj": {"path": "savedModels/HEG/crnn_syn_real_t2.pt", "classes": {0: "Hindi", 1: "English", 2: "Gujarati"}}
}


def load_model(model_path, num_classes):


    crnn = CRNN(3, 32, 64, num_classes, map_to_seq_hidden=32, rnn_hidden=256, leaky_relu=False)
    crnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    crnn.eval()
    return crnn




@app.post("/predict_image/{model_key}")
async def predict_image(model_key: str, file: UploadFile = File(...)):
    if model_key not in model_info:
        return {"error": f"Model key '{model_key}' not found."}

    model_path = model_info[model_key]["path"]
    class_map = model_info[model_key]["classes"]

    try:
        crnn = load_model(model_path, len(class_map))

        image = Image.open(file.file)
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits_seq = crnn(image)  
            logits = torch.mean(logits_seq, dim=0)  
            prediction = torch.argmax(logits, dim=1).item()  


        predicted_label = class_map[prediction]

        return {"predicted_label": predicted_label, "filename": file.filename}

    except Exception as e:
        return {"error": str(e)}



@app.get("/")
def root():
    return {"Hello":"World"}    


