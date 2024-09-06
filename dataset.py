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
    mask = np.zeros(len(src), dtype=int)

    # Populate the mask with 1 for QRS and 2 for P
    for _, row in lbl.iterrows():
        if row['Value'] == 'QRS':
            mask[row['ROILimits_1']:row['ROILimits_2'] + 1] = 1
        elif row['Value'] == 'P':
            mask[row['ROILimits_1']:row['ROILimits_2'] + 1] = 2
        

    # Now we have the mask, we can save it to a CSV file
    mask_df = pd.DataFrame(mask, columns=['mask'])
    X = src.values.flatten()
    Y = mask_df.values.flatten()

    return ECGDataset(X, Y)

def dataset_folder(root):
    files = os.listdir(root)
    subjects = [x.split('_')[0] for x in files]
    subjects = list(set(subjects))
    datasets = []
    for s in subjects:
        src_csv = os.path.join(root, f'{s}_src.csv')
        lbl_csv = os.path.join(root, f'{s}_lbl.csv')
        datasets.append(read_dataset_from_csv(src_csv, lbl_csv))

    return ConcatDataset(datasets)

