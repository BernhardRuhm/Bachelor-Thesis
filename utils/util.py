from argparse import ArgumentParser
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

from datasets import DATASETS_DICT
from augmentation import get_augmentation_type, augment_data

current_dir = os.path.dirname(os.path.abspath(__file__))
archiv_dir = os.path.join(current_dir, "../datasets/UCRArchive_2018/")
models_dir = os.path.join(current_dir,"../models")

def get_data(name):
    if not os.path.isfile(name):
        raise FileNotFoundError("File %s doesn't exist" % name)

    data = np.loadtxt(name, delimiter="\t")
    y_train = data[:, 0] 
    X_train = data[:, 1:]
    return X_train, y_train

def load_dataset(name, 
                 positional_encoding=False, 
                 data_augmentation=False,
                 normalized=False):
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

    X_train, y_train = get_data(file_name + "_TRAIN.tsv")   
    X_test, y_test = get_data(file_name + "_TEST.tsv")   

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

    # Add augmented data

    if data_augmentation:

        target_samples = int(len(X_test) / 0.4)
        additional_train_samples = target_samples - len(X_test) - len(X_train)

        if additional_train_samples > 0:
            augmentation_type = get_augmentation_type(DATASETS_DICT[name]["type"])
            X_train, y_train = augment_data(X_train, y_train, additional_train_samples, augmentation_type)


    # Manually split data, otherwise pre-splits are used
    # if custom_split_ratio > 0.:
    #     X_combined = np.concatenate((X_train, X_test), axis=0)
    #     y_combined = np.concatenate((y_train, y_test), axis=0)
    #     X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=custom_split_ratio, shuffle=True, random_state=1)

    # Add additional positional encoding features
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
    all_datasets = list(DATASETS_DICT.keys()) 
    return sorted(all_datasets)

def get_testing_datasets():
    datasets = ['Crop', 'NonInvasiveFetalECGThorax1', 'Fungi', 'Strawberry', 'TwoPatterns', 
                'MelbournePedestrian', 'UWaveGestureLibraryAll', 'InsectEPGRegularTrain', 
                'ElectricDevices', 'EOGHorizontalSignal', 'FordB', 'InsectWingbeatSound', 
                'PigAirwayPressure', 'Worms']

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

def arg_parser():

    parser = ArgumentParser()

    # model specs
    parser.add_argument('--model', type=str, default="LSTM", help='Model to be trained')
    parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units per LSTM layer')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of sequential LSTM cells')
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6], 
                        help='Normalization used after each LSTM layer.' 
                        '0: no normalization' 
                        '1: Batchnorm1d with affine False' 
                        '2: Batchnorm1d with affine True'
                        '3: Post Layernorm'
                        '4: Pre Layernorm')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout ratio for LSTM Layers except last one')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay ratio')

    # training specs
    parser.add_argument('--device', type=str, default='cuda:0', help='Selected device')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Train and validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='(initial) learning rate for training')

    # input manipulation
    parser.add_argument('--positional_encoding', default=False, action='store_true', help='Defines if positional encoding is added to the input features')
    parser.add_argument('--data_augmentation', default=False, action='store_true', help='add addition augmented data for 70 / 30 split') 
    parser.add_argument('--custom_split', type=float, default=0., help='ratio of test split, after shuffling default train and test sets')
        
    #util
    parser.add_argument('--export', default=False, action='store_true', help='Export model as .onnx')
    parser.add_argument('--confusionflow', default=False, action='store_true', help='Use confusion flow to log data')
    parser.add_argument('--framework', default='keras', type=str, help="Framework to be used")

    return parser

