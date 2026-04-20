import os
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from model import get_resnet_model
import config
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np

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

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store probabilities for the positive class (Pneumonia -> index 1)
            all_probs.extend(probs[:, 1].cpu().numpy())

    test_epoch_loss = test_loss / len(test_loader.dataset)
    test_epoch_acc = test_corrects.double() / len(test_loader.dataset)

    print(f"Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))
    
    # --- Scientific Charts Generation ---
    metrics_dir = os.path.join(config.DATA_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="pastel")

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
                annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    cm_path = os.path.join(metrics_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    roc_path = os.path.join(metrics_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")

if __name__ == "__main__":
    evaluate_model()
