# Global Variables
import torch

EPOCHS = 50
DEBUG = False # Toggle this to only run for 1% of the training data
ENABLE_GPU = True  # Toggle this to enable or disable GPU
BATCH_SIZE = 16
SOFTMAX = True
TRAIN_MC_DROPOUT = False
SAMPLES = 3
FORWARD_PASSES = 100
BBB = True
LOAD = False
LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
SAVE_DIR = "saved_models"
ISIC_pred = False
TRAIN = True
NUM_MODELS = 1
IMAGE_SIZE = 224
DEVICE = torch.device("cuda")