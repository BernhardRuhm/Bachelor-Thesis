import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append("../utils")
from util import load_dataset, extract_metrics, archiv_dir, embed_positional_features 


class UCRDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_Dataloaders(dataset, batch_size, positional_encoding):

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset, positional_encoding)
    print(X_train.shape)
    seq_len, input_dim, n_classes = extract_metrics(X_train, y_train)

    metrics = (seq_len, input_dim, n_classes)

    dl_train = DataLoader(UCRDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(UCRDataset(X_test, y_test), batch_size=batch_size, shuffle=True)

    return dl_train, dl_test, metrics
