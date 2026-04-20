import os
import io
import base64
import time
import uuid
import numpy as np
import torch
import aiosqlite
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
from PIL import Image
from fpdf import FPDF

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

# Directories
DB_PATH = os.path.join(config.DATA_DIR, "insightxray.db")
HISTORY_IMG_DIR = os.path.join(config.DATA_DIR, "history_images")
os.makedirs(HISTORY_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(config.DATA_DIR, "metrics"), exist_ok=True)

# Global model state
ml_models = {}

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                prediction TEXT,
                confidence REAL,
                image_path TEXT,
                heatmap_path TEXT
            )
        """)
        await db.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init Database
    await init_db()
    
    # Load PyTorch Model on Startup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading Fast Inference Server on {device}...")
    model = get_resnet_model(pretrained=False)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Init target layer for Grad-CAM
    target_layers = [model.layer4[-1]]
    
    ml_models["model"] = model
    ml_models["device"] = device
    ml_models["target_layers"] = target_layers
    yield
    ml_models.clear()

app = FastAPI(title="InsightXray AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_and_read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        data = list(image.getdata())
        image_clean = Image.new(image.mode, image.size)
        image_clean.putdata(data)
        return image_clean
    except Exception:
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
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = torch.max(output, 1)
        
    predicted_class = config.CLASS_NAMES[pred.item()]
    confidence = probabilities[pred.item()].item() * 100

    # ----- Grad-CAM (XAI) ----- #
    torch.set_grad_enabled(True)
    model.eval()
    cam_tensor = input_tensor.clone().requires_grad_(True)
    cam = GradCAM(model=model, target_layers=ml_models["target_layers"])
    grayscale_cam = cam(input_tensor=cam_tensor, targets=[ClassifierOutputTarget(pred.item())])[0, :]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    viz_tensor = np.clip(std * input_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + mean, 0, 1)
    visualization = show_cam_on_image(viz_tensor, grayscale_cam, use_rgb=True)
    
    heatmap_img = Image.fromarray(visualization)
    buffer = io.BytesIO()
    heatmap_img.save(buffer, format="JPEG")
    cam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    torch.set_grad_enabled(False)

    # --- Save to History ---
    entry_id = str(uuid.uuid4())
    img_filename = f"{entry_id}_orig.jpg"
    heatmap_filename = f"{entry_id}_heat.jpg"
    
    img_save_path = os.path.join(HISTORY_IMG_DIR, img_filename)
    heatmap_save_path = os.path.join(HISTORY_IMG_DIR, heatmap_filename)
    
    clean_img.save(img_save_path)
    heatmap_img.save(heatmap_save_path)
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO history (id, timestamp, prediction, confidence, image_path, heatmap_path) VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, time.time(), predicted_class, round(confidence, 2), img_filename, heatmap_filename)
        )
        await db.commit()

    return JSONResponse(content={
        "id": entry_id,
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": f"data:image/jpeg;base64,{cam_b64}"
    })

@app.get("/api/history")
async def get_history():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 50") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

@app.get("/api/report/{entry_id}")
async def generate_report(entry_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM history WHERE id = ?", (entry_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Entry not found")
            
            # Generate PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Helvetica", "B", 24)
            pdf.set_text_color(0, 85, 113) # InsightXray Brand Color
            pdf.cell(0, 20, "InsightXray: Diagnostic Report", ln=True, align="C")
            
            pdf.set_font("Helvetica", "", 12)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 10, f"Report ID: {row['id']}", ln=True, align="C")
            pdf.cell(0, 10, f"Date: {time.ctime(row['timestamp'])}", ln=True, align="C")
            pdf.ln(10)
            
            # Prediction Results
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, "1. Diagnosis Executive Summary", ln=True)
            
            color = (239, 68, 68) if row['prediction'] == "PNEUMONIA" else (16, 185, 129)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(*color)
            pdf.cell(0, 10, f"PREDICTION: {row['prediction']}", ln=True)
            
            pdf.set_font("Helvetica", "", 14)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 10, f"Confidence Level: {row['confidence']}%", ln=True)
            pdf.ln(10)
            
            # XAI Reasoning
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, "2. AI Reasoning (Grad-CAM Heatmap)", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 8, "The heatmap overlay below highlights clinical features (opacities, consolidations) within the pulmonary region that informed the model's high-confidence prediction.")
            pdf.ln(5)
            
            # Embedded Images
            img_path = os.path.join(HISTORY_IMG_DIR, row['image_path'])
            heat_path = os.path.join(HISTORY_IMG_DIR, row['heatmap_path'])
            
            pdf.image(img_path, x=15, y=pdf.get_y(), w=85)
            pdf.image(heat_path, x=110, y=pdf.get_y(), w=85)
            
            pdf.output("temp_report.pdf")
            with open("temp_report.pdf", "rb") as f:
                pdf_bytes = f.read()
            
            return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=InsightXray_Report_{entry_id}.pdf"})

# Mount Folders
app.mount("/metrics", StaticFiles(directory=os.path.join(config.DATA_DIR, "metrics")), name="metrics")
app.mount("/history_images", StaticFiles(directory=HISTORY_IMG_DIR), name="history_images")
app.mount("/", StaticFiles(directory="serving/frontend", html=True), name="frontend")
