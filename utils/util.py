import os 
import csv

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

import torch
import onnx
from onnxsim import simplify

current_dir = os.path.dirname(os.path.abspath(__file__))
archiv_dir = os.path.join(current_dir, "../datasets/UCR_TS_Archive_2015/")
models_dir = os.path.join(current_dir,"../models")

def get_data(name):
    if not os.path.isfile(name):
        raise FileNotFoundError("File %s doesn't exist" % name)

    data = np.loadtxt(name, delimiter=",")
    y_train = data[:, 0] 
    X_train = data[:, 1:]
    return X_train, y_train

def load_dataset(name, positional_encoding=False, normalized=False, custom_split=False, augmentation=None):
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

    # scale data between -1 and 1
    if normalized:
        X_train_max = X_train.max()
        X_train_min = X_train.min()

        X_train = 2. * (X_train - X_train_min) / (X_train_max - X_train_min) - 1
        X_test = 2. * (X_test - X_train_min) / (X_train_max - X_train_min) - 1

    # if univariate, feature dimension is added
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

    # Add additional positional encoding features
    if positional_encoding:
        X_train = embed_positional_features(X_train)
        X_test = embed_positional_features(X_test)

    # Manually split data, otherwise pre-splits are used
    if custom_split:
        X_combined = np.concatenate((X_train, X_test), axis=0)
        y_combined = np.concatenate((y_train, y_test), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, shuffle=True, random_state=1)


    if augmentation == 'jitter':
        X_train = jitter(X_train)
    elif augmentation == 'window_warp':
        X_train = window_warp(X_train)

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
    df = pd.DataFrame(data=np.zeros((0, 4)), 
                      columns = ["dataset", "accuracy", "peak accuracy", "train time"])
    df.to_csv(file_name, index=False, header=True)

def add_results(file_name, dataset, acc, peak_acc,time):
    new_results = {"dataset": [dataset], "accuracy": [acc], "peak accuracy": [peak_acc], "train time": [time]}
    df = pd.DataFrame(new_results)
    df.to_csv(file_name, mode="a", index=False, header=False)

def create_training_csv(file_name):
    df = pd.DataFrame(data=np.zeros((0, 6)),
                      columns = ["epoch", "train loss", "val loss", "train acc", "val acc", "lr"])
    df.to_csv(file_name, index=False, header=True)

def create_predictions_csv(file_name):
    df = pd.DataFrame(data=np.zeros((0, 2)), index=[],
                      columns = ["epoch", "predictions"])
    df.to_csv(file_name, index=False, header=True)


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

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i],dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
            end_seg = pat[window_ends[i]:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:,dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def get_datasets_hiddensize(file_name):
    content = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            dataset = row[0]
            hidden_size = row[1]
            content.append((dataset, hidden_size))
    return content 


def simplify_model(model_file):

    onnx_model = onnx.load(model_file)
    onnx_simplified, _ = simplify(onnx_model)

    graph = onnx_simplified.graph
    for node in graph.node:
        if node.op_type == "LSTM":
            # remove state outputs
            del node.output[1]
            del node.output[1]

    onnx.save(onnx_simplified, model_file)

def export_model(model_checkpoint, model_name, dataset, seq_len, input_dim, device):
    
    model = torch.load(model_checkpoint)
    # export model to .onnx
    dummy_input = torch.randn(seq_len, 1, input_dim).to(device)

    model_file = os.path.join(models_dir, model_name + "_" + dataset + ".onnx")
    onnx_model = torch.onnx.export(model, 
                                     dummy_input, 
                                     model_file,
                                     export_params=True,
                                     input_names =  ["input"],
                                     output_names =  ["output"])

    if simplify:
        simplify_model(model_file)
