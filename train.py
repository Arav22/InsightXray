import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from model import get_resnet_model
import config

def train_model():
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure checkpoint dir exists
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load data
    train_loader, val_loader, _, classes = get_dataloaders()
    
    # Load model
    model = get_resnet_model(pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save Best Model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"Best model saved with Validation Accuracy: {val_epoch_acc:.4f}\n")
        else:
            print("\n")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()
