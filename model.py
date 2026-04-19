import torch
import torch.nn as nn
from torchvision import models
import config

def get_resnet_model(pretrained=True):
    """Loads a ResNet18 model and replaces the final fully connected layer."""
    
    # Load the pretrained ResNet18 model
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze the earlier layers if you only want to train the final few
    # For fine-tuning medical images, it's sometimes better to train everything with a small LR
    # but we'll freeze the initial layers for slightly faster convergence as a default approach
    for param in model.parameters():
        param.requires_grad = True # Here we choose to fine-tune the whole network

    # Replace the final fully connected layer to match the number of classes (2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # Add some dropout for regularization
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )
    
    return model

if __name__ == "__main__":
    # Test the model structure
    model = get_resnet_model()
    print(model)
    
    # Dummy forward pass
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") 
