import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_transforms():
    """Returns training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_dataloaders():
    """Creates and returns DataLoaders for train, val, and test splits."""
    train_transform, val_test_transform = get_transforms()

    # Load datasets
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=val_test_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes
