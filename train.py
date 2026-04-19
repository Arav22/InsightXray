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
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    # Load model
    model = get_resnet_model(pretrained=True)
    model = model.to(device)

    # Loss and optimizer (Added Weight Decay for L2 Regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler: Drops learning rate if validation accuracy stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOPPING_PATIENCE = 4

    print("Starting training with Hyperparameter Optimization features...")
    for epoch in range(config.NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Step the scheduler
        scheduler.step(val_epoch_acc)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save Best Model and check Early Stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"🌟 Best model updated (Val Acc: {val_epoch_acc:.4f})\n")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Early Stopping Counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}\n")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping triggered!")
                break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
    
    # ------------------
    # FINAL TEST EVALUATION
    # ------------------
    print("\nLoading Best Model for Final Test Set Evaluation...")
    best_model = get_resnet_model(pretrained=False)
    
    # Load correctly onto the right device
    best_model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
    best_model = best_model.to(device)
    best_model.eval()
    
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            
    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f"\n🎉 OFFICIAL TEST DATA ACCURACY: {test_acc:.4f}\n")

if __name__ == "__main__":
    train_model()
