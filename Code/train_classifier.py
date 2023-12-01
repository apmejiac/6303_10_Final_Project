#### Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import torchvision
import torchvision.transforms

OR_PATH = os.getcwd()
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH) # Come back to the directory where the code resides , all files will be left on this directory

n_epoch = 15
BATCH_SIZE = 100
LR = 0.001 # 0.0001 with 8 epochs, 100 batch, 3 channel and 250 img, 0.55 thrshold: 0.50357

## Image processing
CHANNELS = 3
IMAGE_SIZE = 300

# Create Dataframe for Image Processing
def create_dataframe():
    df = pd.DataFrame({"id": None, "label": None})
    
    # For file in filepath
    # Iterate through all images
    