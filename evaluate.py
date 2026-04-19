import os
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from model import get_resnet_model
import config
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model():
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, classes = get_dataloaders()
    
    # Load model
    model = get_resnet_model(pretrained=False)
    
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"No trained model found at {config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_epoch_loss = test_loss / len(test_loader.dataset)
    test_epoch_acc = test_corrects.double() / len(test_loader.dataset)

    print(f"Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model()
