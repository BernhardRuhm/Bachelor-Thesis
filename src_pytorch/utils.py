import numpy as np
import os

from aeon.datasets import load_from_tsfile

import torch

from models import LSTMDense

archiv_dir = "./datasets/Univariate_ts/"


def load_dataset(name, return_type="np"):
    """
    Loads a UCR dataset

    Args:
        name: name of dataset to be loaded
    Returns:
        Tuple of shape (X_train, y_train), (X_test, y_test)
    """
    
    path = os.path.join(archiv_dir, name)

    if not os.path.exists(path):
        raise FileNotFoundError("Dataset directory %s doesn't exist" % name)
    
    train_tsv = os.path.join(path, name + "_TRAIN.ts")
    if not os.path.isfile(train_tsv):
        raise FileNotFoundError("File %s doesn't exist" % train_tsv)

    test_tsv = os.path.join(path, name + "_TEST.ts")
    if not os.path.isfile(test_tsv):
        raise FileNotFoundError("File %s doesn't exist" % test_tsv)
    X_train, y_train = load_from_tsfile(train_tsv)
    X_test, y_test = load_from_tsfile(test_tsv)
        
    # reshape to (samples, seq_len, input_dim)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    y_train = transform_labels(y_train)
    y_test = transform_labels(y_test)
    
    return (X_train, y_train), (X_test, y_test) 


def transform_labels(y):
    """
    Transforms lables, so that they begin with 0 

    Args:
        y: lables that should be transformed
    Return:
        transformed labels
    """
    unique_values = np.unique(y)
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    transformed_y = np.array([value_to_index[value] for value in y])

    return transformed_y

def extract_metrics(X, y):
    """
    Returns meta information from time series

    Args:
        X: Time Series Dataset
        y: Classes
    Returns:
        Tuple of shape (sequence length, input dimension, number of classes)
    """
    _, seq_len, dim = X.shape
    classes = len(np.unique(y))
    return seq_len, dim, classes


def train_pytorch_model():

    return None

def train_eval_loop(n_hidden=32, n_layers=2, batch_size=128, n_epochs=250):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    datasets = ["HandOutlines"]


    for ds in datasets:

        (X_train, y_train), (X_test, y_test) = load_dataset(ds)
        seq_len, input_dim, n_classes = extract_metrics(X_train, y_train)

        model_nn = LSTMDense(seq_len, input_dim, n_hidden, n_layers, n_classes)

        model = torch.jit.script(model_nn)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for e in range(n_epochs):
            model.train()
            for b in range(0, len(X_train), batch_size):
                x_sample = X_train[b:b+batch_size,:,:]
                y_sample = y_train[b:b+batch_size]

                x_batch = torch.tensor(x_sample, dtype=torch.float32, device=device)
                y_batch = torch.tensor(y_sample, dtype=torch.long, device=device)

                # model.init_states(x_batch.size(0))

                out = model(x_batch)
                # print(model.graph_for(x_batch))
                # return
                loss = criterion(out, y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0

                for b in range(0, len(X_test), batch_size):
                    x_sample = X_test[b:b+batch_size,:,:]
                    y_sample = y_test[b:b+batch_size]

                    x_batch = torch.tensor(x_sample, dtype=torch.float32, device=device)
                    y_batch = torch.tensor(y_sample, dtype=torch.long, device=device)

                    # model.init_states(x_batch.size(0))

                    out = model(x_batch)
                    _, pred = torch.max(out,1)
                    correct += (pred == y_batch).sum().item()
                    total += len(y_batch)
                
                acc = correct / total
                print("step: ", e, "train loss:", loss.item(), "val acc:", acc, )




        

