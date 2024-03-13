import os 
import csv

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

current_dir = os.path.dirname(os.path.abspath(__file__))
archiv_dir = os.path.join(current_dir, "../datasets/UCR_TS_Archive_2015/")
models_dir = os.path.join(current_dir,"../models")
result_dir = os.path.join(current_dir, "../results")

def get_data(name):
    if not os.path.isfile(name):
        raise FileNotFoundError("File %s doesn't exist" % train_tsv)

    data = np.loadtxt(name, delimiter=",")
    y_train = data[:, 0] 
    X_train = data[:, 1:]
    return X_train, y_train

def load_dataset(name, positional_encoding=False):
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
    
    file_name = os.path.join(path, name)

    X_train, y_train = get_data(file_name + "_TRAIN")   
    X_test, y_test = get_data(file_name + "_TEST")   

    # transform labels to start with 0
    y_train = transform_labels(y_train)
    y_test = transform_labels(y_test)

    # if univariate, feature dimension is added
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

    if positional_encoding:
        X_train = embed_positional_features(X_train)
        X_test = embed_positional_features(X_test)
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

def get_all_datasets():
    all_datasets = os.listdir(archiv_dir)  
    datasets = []
    for ds in all_datasets:
        (X_train, y_train), _ = load_dataset(ds) 
        if X_train.shape[0] > 400:
            datasets.append(ds)
    return sorted(datasets)

def create_results_csv(file_name):
    df = pd.DataFrame(data=np.zeros((0, 5)), index=[],
                      columns = ["dataset", "accuracy", "precision", "recall", "train time"])

    df.to_csv(file_name, index=False)

def add_results(file_name, dataset, acc, prec, recall, time):
    new_results = {"dataset": [dataset], "accuracy": [acc], "precision": [prec], 
                   "recall": [recall], "train time": [time]}
    df = pd.DataFrame(new_results)
    df.to_csv(file_name, mode="a", index=False, header=False)

def calculate_eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    return acc, prec, recall

def embed_positional_features(data):
    num_samples = data.shape[0]
    seq_len = data.shape[1]

    positional_features = np.zeros((num_samples, seq_len, 2)) 
    positional_features[:, :, 0] = (np.sin(2*np.pi/seq_len * np.arange(seq_len)) + 1) * 0.5 
    positional_features[:, :, 1] = (np.cos(2*np.pi/seq_len * np.arange(seq_len)) + 1) * 0.5

    return np.concatenate((data, positional_features), axis=-1) 

def get_datasets_hiddensize(file_name):
    content = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            dataset = row[0]
            hidden_size = row[1]
            content.append((dataset, hidden_size))
    return content 
