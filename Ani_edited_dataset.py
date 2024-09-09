import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat


class ECGDataset(Dataset):
    def __init__(self, X, Y, seqlen=50):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.seqlen = seqlen
        self.num_slices = len(self.X) // seqlen

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        start_idx = idx * self.seqlen
        end_idx = start_idx + self.seqlen
        inputs = self.X[start_idx:end_idx].unsqueeze(-1)
        # inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
        return inputs, self.Y[start_idx:end_idx]
    
def read_x_from_csv(csv):
    X = pd.read_csv(csv)
    X = X.Channel0.ffill().bfill()*-1
    X = X.values
    Y = np.zeros(len(X))

    return ECGDataset(X, Y)

def read_x_from_mat(mat_file):
    # Load data from .mat file
    mat_data = loadmat(mat_file)
    
    # Assume 'Channel0' contains the ECG data; adjust if necessary
    X = mat_data['ecg'].squeeze()  # Squeeze in case it's a 2D array of shape (n,1)
    
    # Perform any necessary preprocessing
    X = pd.Series(X).ffill().bfill() * -1
    X = X.values
    Y = np.zeros(len(X))  # Assuming Y is needed as a zero array of the same length

    return ECGDataset(X, Y)


# ANI MODIFY THIS!!!!!!!!!!!!! Since you have more labels than just QRS and P
def read_dataset_from_csv(train_csv, label_csv):
    src = pd.read_csv(train_csv)
    lbl = pd.read_csv(label_csv)
    
    num_classes = 6  # Number of classes for one-hot encoding
    mask = np.zeros((len(src), num_classes), dtype=int)

    # Map label values to class indices
    label_map = {
        'QRS-Complex': 3,
        'P-Wave': 1,
        'PR-Interval': 2,
        'RR-Interval': 5,
        'QT-Interval': 4
    }

    # Populate the mask with one-hot encoding
    for _, row in lbl.iterrows():
        ROI1 = row['ROILimits_1']
        ROI1 = int(ROI1 * 512)
        ROI2 = row['ROILimits_2']
        ROI2 = int(ROI2 * 512)
        label_index = label_map.get(row['Value'], None)
        if label_index is not None:
            mask[ROI1:ROI2 + 1, label_index] = 1

    # Flatten src and mask arrays
    X = src.values.flatten()
    Y = mask.reshape(-1, num_classes)  # Flatten mask for DataLoader

    return ECGDataset(X, Y)

def dataset_folder(root):
    files = os.listdir(root)
    subjects = [x[:x.rfind('_')] for x in files if x.endswith('.csv')]
    subjects = list(set(subjects))
    datasets = []
    for s in subjects:
        src_csv = os.path.join(root, f'{s}_src.csv')
        lbl_csv = os.path.join(root, f'{s}_lbl.csv')
        datasets.append(read_dataset_from_csv(src_csv, lbl_csv))

    return ConcatDataset(datasets)

