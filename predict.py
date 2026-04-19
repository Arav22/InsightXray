import os
import torch
import sys
from PIL import Image
from model import get_resnet_model
from data_loader import get_transforms
import config

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = get_resnet_model(pretrained=False)
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"No trained model found at {config.BEST_MODEL_PATH}. Cannot predict.")
        return

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Get transform
    _, val_test_transform = get_transforms()

    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = val_test_transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = torch.max(output, 1)

    predicted_class = config.CLASS_NAMES[pred.item()]
    confidence = probabilities[pred.item()].item() * 100

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: Normal: {probabilities[0].item()*100:.2f}%, Pneumonia: {probabilities[1].item()*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
