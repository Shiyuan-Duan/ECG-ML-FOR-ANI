import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import ECGDataset, read_dataset_from_csv, dataset_folder
from sklearn.metrics import accuracy_score
from torchvision import transforms


def smooth_predictions(preds, window_size=10):
    smoothed_array = np.copy(preds)
    for i in range(len(preds) - window_size + 1):
        # Define the current window
        window = preds[i:i + window_size]
        # Check if the first and last elements of the window are the same
        if window[0] == window[-1] and window[0] != 0:
            # Set all elements in the window in the smoothed array to this value
            smoothed_array[i:i + window_size] = window[0]
    
    return smoothed_array



