import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '.data')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model saving
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Image parameters
IMAGE_SIZE = (224, 224)  # ResNet expects 224x224
NUM_CLASSES = 2  # Normal and Pneumonia

# Classes
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
