from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io, base64

from app.model_loader import get_model
from app.inference import predict

# Initialize FastAPI
app = FastAPI(title="WAFFNet++ Script Identification API", version="1.0")

# Load model
model, device = get_model()

# Templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict/ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    label = predict(model, device, image)

    # Convert image to base64 to display on page
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "prediction": label, "image_data": img_str}
    )

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    label = predict(model, device, image)
    return {"predicted_language": label}
