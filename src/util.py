import os 
import numpy as np
from aeon.datasets import load_from_tsv_file, load_from_tsfile

archiv_dir = "./datasets/Univariate_ts/"
models_dir = "./models/"
log_dir = "./logs/"

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

