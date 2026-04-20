import os
import io
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image

# Import ML code properly from parent directory
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model import get_resnet_model
from data_loader import get_transforms
import config

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Global model state
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load PyTorch Model on Startup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Fast Inference Server on {device}...")
    model = get_resnet_model(pretrained=False)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Init target layer for Grad-CAM (ResNet18 deepest convolutional layer)
    target_layers = [model.layer4[-1]]
    
    ml_models["model"] = model
    ml_models["device"] = device
    ml_models["target_layers"] = target_layers
    yield
    # Cleanup memory
    ml_models.clear()

app = FastAPI(title="InsightXray AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_and_read_image(file_bytes: bytes) -> Image.Image:
    """Strictly validates image integrity and strips EXIF payloads."""
    try:
        # Verify block
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  
        # Reload after verification passes
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        # Brutally strip EXIF and hidden payloads by copying raw data into a new container
        data = list(image.getdata())
        image_clean = Image.new(image.mode, image.size)
        image_clean.putdata(data)
        return image_clean
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid or malicious image file.")

@app.post("/api/predict")
async def predict_xray(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty payload.")
    
    clean_img = sanitize_and_read_image(contents)
    
    model = ml_models["model"]
    device = ml_models["device"]
    _, val_test_transform = get_transforms()
    
    input_tensor = val_test_transform(clean_img).unsqueeze(0).to(device)
    
    # Predict normally first
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = torch.max(output, 1)
        
    predicted_class = config.CLASS_NAMES[pred.item()]
    confidence = probabilities[pred.item()].item() * 100

    # ----- EXPLAINABLE AI (Grad-CAM) ----- #
    torch.set_grad_enabled(True)
    model.eval()
    
    # Required for the backward pass inside Grad-CAM
    cam_tensor = input_tensor.clone().requires_grad_(True)
    
    cam = GradCAM(model=model, target_layers=ml_models["target_layers"])
    targets = [ClassifierOutputTarget(pred.item())]
    grayscale_cam = cam(input_tensor=cam_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare base image for overlay (Reverse the PyTorch Normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    viz_tensor = input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    viz_tensor = std * viz_tensor + mean
    viz_tensor = np.clip(viz_tensor, 0, 1)
    
    # Overlay the Heatmap mathematically
    visualization = show_cam_on_image(viz_tensor, grayscale_cam, use_rgb=True)
    
    # Cast to base64 JPEG to beam securely to the Frontend
    cam_img = Image.fromarray(visualization)
    buffer = io.BytesIO()
    cam_img.save(buffer, format="JPEG", quality=85)
    cam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    torch.set_grad_enabled(False)

    return JSONResponse(content={
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": f"data:image/jpeg;base64,{cam_b64}"
    })

# Ensure metrics directory exists to prevent server crash on cold-boot
os.makedirs(".data/metrics", exist_ok=True)

# Serve Metric Charts strictly statically
app.mount("/metrics", StaticFiles(directory=".data/metrics"), name="metrics")

# Mount Phase 3 Frontend Layer
app.mount("/", StaticFiles(directory="serving/frontend", html=True), name="frontend")
