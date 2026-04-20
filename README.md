# 🫁 InsightXray: Advanced Pneumonia Diagnostic Panel

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**InsightXray** is a production-grade medical imaging AI designed to detect Pneumonia from chest X-rays with high precision. By leveraging **Deep Learning (ResNet18)** and **Explainable AI (Grad-CAM)**, the system provides both a diagnosis and visual evidence for its decision-making process.

---

## 🚀 Key Features

*   **93.2% Test Accuracy**: Rigorously trained on thousands of clinical X-ray images.
*   **Explainable AI (XAI)**: Integrated Grad-CAM heatmaps highlight exactly which regions of the lungs triggered the diagnosis.
*   **Production-Grade Architecture**: Modularized into a high-performance FastAPI microservice.
*   **iOS 26 Aesthetic UI**: A modern, glassmorphism-based web dashboard designed for medical professionals.
*   **Scientifically Verified**: Automatic generation of Confusion Matrices and ROC Curves to prove model validity.
*   **Secure & Sanitized**: Strict image interrogation and EXIF stripping to prevent malicious payloads.

---

## 🎨 Dashboards & Metrics

### 🖥️ The Interactive UI
*Features a premium frosted-glass design with real-time confidence tracking and XAI toggles.*

![UI Screenshot](https://via.placeholder.com/800x400.png?text=UI+Dashboard+Preview)

### 📊 Scientific Validation
*Our model is proven on unseen test sets.*

| Confusion Matrix | ROC Curve |
| :---: | :---: |
| ![CM](.data/metrics/confusion_matrix.png) | ![ROC](.data/metrics/roc_curve.png) |

---

## 🛠️ Technical Architecture

### The Brain: ResNet18
We utilize a ResNet18 architecture, fine-tuned for binary classification (`NORMAL` vs `PNEUMONIA`). 
*   **Optimizer**: Adam with L2 Regularization (`weight_decay=1e-4`).
*   **Scheduler**: `ReduceLROnPlateau` for hyper-fine learning rate adjustments.
*   **Input**: Normalized 224x224 RGB tensors.

### The Service: FastAPI + Grad-CAM
The serving layer uses **FastAPI** for low-latency inference. Explainability is provided by **Grad-CAM**, which analyzes the final convolutional layer to produce activation heatmaps.

---

## 📦 Local Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed (the fastest Python package manager).

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/InsightXray.git
    cd InsightXray
    ```

2.  **Initialize Environment & Dependencies**
    ```bash
    uv sync
    ```

3.  **Run the Web Dashboard**
    ```bash
    uv run uvicorn serving.api:app --reload
    ```
    *Access at http://127.0.0.1:8000*

---

## 🐳 Docker Deployment

To build a secure, lightweight production container:

```bash
docker build -t insight-xray .
docker run -p 8000:8000 insight-xray
```

---

## 📁 Project Structure

*   `serving/`: Web API and Frontend assets.
*   `checkpoints/`: Optimized model weights (`best_model.pth`).
*   `evaluate.py`: Statistical auditing and chart generation.
*   `train.py`: The high-performance training loop.
*   `model.py`: Core CNN architecture definitions.
*   `.data/metrics/`: Generated scientific proof (PNGs).

---

## ⚖️ Disclaimer
*This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
