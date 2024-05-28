import sys
import os
import time 
import json
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import onnx
from onnxsim import simplify


from confusionflow.logging.logfunction import log_epoch
from confusionflow.logging import Fold, Run

from dataloader import get_Dataloaders
from models import valid_models, LSTMFCN, init_weights, generate_model

sys.path.append("../utils")
from util import models_dir, create_results_csv, add_results, export_model, create_predictions_csv, create_training_csv 
from visualize import visualize_training_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# datasets = ['50words', 'ChlorineConcentration', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
# 'ElectricDevices', 'FordA', 'FordB', 'NonInvasiveFatalECG_Thorax1', 'UWaveGestureLibraryAll'] 
# datasets = ['ChlorineConcentration', 'ElectricDevices', 'FordA', 'UWaveGestureLibraryAll'] 
datasets = ['50words', 'Cricket_X', 'NonInvasiveFatalECG_Thorax1', 'FordA', 'FaceAll', 'PhalangesOutlinesCorrect', 'ShapesAll', 'wafer']
# datasets = ["FaceAll"]

def main():
   
    # model args 
    model_name = args.model
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    batch_norm = args.batch_norm
    dropout = args.dropout
    weight_decay = args.weight_decay

    # training args 
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # input manipulation args
    positional_encoding = args.positional_encoding
    custom_split = args.custom_split
    augmentation_type = args.augmentation_type
    augmentation_ratio = args.augmentation_ratio

    # util args
    export = args.export
    use_confusionflow = args.confusionflow

    # create experiment path
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 
    path_suffix = " HS:" + str(hidden_size) + " NL:" + str(n_layers) + " " + time_stamp
    path_prefix = model_name

    if positional_encoding:
        path_prefix += "_PosEnc"  
    if batch_norm != 0:
        path_prefix += "_BN:" + str(batch_norm)

    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_path, model_name)

    experiment_path = os.path.join('experiments', path_prefix + path_suffix)
    os.makedirs(experiment_path, exist_ok=True) 

    # Write args of experiment to a file
    with open(os.path.join(experiment_path,'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    result_file = os.path.join(experiment_path, 'results.csv')
    create_results_csv(result_file)

    # Iterate through datasets
    for ds in sorted(datasets):

        predictions_file = os.path.join(experiment_path, ds + "_pred.csv") 
        create_predictions_csv(predictions_file)

        training_file = os.path.join(experiment_path, ds + ".csv") 
        create_training_csv(training_file)

        dl_train, dl_test, metrics = get_Dataloaders(ds, batch_size, positional_encoding, custom_split, augmentation_type, augmentation_ratio)
        seq_len, input_dim, n_classes = metrics

        model = generate_model(model_name, device, input_dim, hidden_size, n_classes, n_layers, batch_norm, dropout) 
        model.to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs / 4), eta_min=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # specify confusion flow folds and run
        if use_confusionflow:
            dl_train_log, dl_test_log, _ = get_Dataloaders(ds, batch_size, positional_encoding, custom_split, augmentation_type, augmentation_ratio)
            train_fold = Fold(data=dl_train_log, foldId='ED_train', dataset_config='../datasets/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices.yml')
            test_fold = Fold(data=dl_test_log, foldId='ED_test', dataset_config='../datasets/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices.yml')
            run = Run(runId='example_confusionflow', folds=[train_fold, test_fold], trainfoldId='ED_train')

        best_train_loss = 1e3
        best_val_acc = 0 
        peak_val_acc = 0

        start_time = time.time()
        print("Training: %s %s" % (model_name, ds))

        for e in range(epochs):

            if use_confusionflow:
                log_epoch(run, model, device, e, numclass=n_classes)

            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, dl_train)            
            val_loss, val_acc, predictions = evaluate(model, criterion, dl_test)
            scheduler.step()

            if e % 10 == 0:
                print("Epoch:", e, "train_loss:", train_loss, "val_acc:", val_acc, "lr:", get_lr(optimizer), "\n")

            training_data = [{'epoch': e, 'train loss': train_loss, 'val loss': val_loss,
                           'train acc': train_acc, 'val acc': val_acc, 'lr': get_lr(optimizer)}]

            # save training and validation data per epoch
            pd.DataFrame(training_data).to_csv(os.path.join(training_file), mode='a', header=False, index=False)
            # save predictions per epoch
            predictions_str = ','.join(map(str, predictions))  # Assuming predictions are floats 
            pd.DataFrame({'epoch': [e], 'predictions': [predictions_str]}).to_csv(predictions_file, mode='a', header=False, index=False)

            if train_loss < best_train_loss:
                best_train_loss = train_loss        
                best_val_acc = val_acc
                torch.save(model, checkpoint_path)
            
            if val_acc > peak_val_acc:
                peak_val_acc = val_acc

        train_time = time.time() -  start_time


        add_results(result_file, ds, best_val_acc, peak_val_acc, train_time)

        if export:
            export_model(checkpoint_path, model_name, ds, seq_len, input_dim, device)

        if use_confusionflow:
            run.export(logdir='confusionflow_logs')

def get_lr(optimizer):
    """
    Return current learning rate of optimizer
    optimizer: torch optimizer of which the lr should be returned
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(model, optimizer, criterion, dataloader):
    correct = 0
    total_samples = 0
    loss_total = 0
    model.train()
    for i, (x_batch, y_batch) in enumerate(dataloader):
        # LSTM is trained batch_first False
        x_batch = torch.swapaxes(x_batch, 0, 1).to(device)
        y_batch = y_batch.to(device)
        # print("input shape", x_batch.shape)

        optimizer.zero_grad()

        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        # running loss
        loss_total += loss.item()

        # calculate correct predictions of the batch
        _, pred = torch.max(out,1)
        correct += (pred == y_batch).sum().item()
        total_samples += len(y_batch)

    # calculate total average train loss and train accuracy
    train_loss = loss_total/ (i+1)
    train_acc = correct / total_samples

    return train_loss, train_acc

def evaluate(model, criterion, dataloader):
    correct = 0
    total_samples = 0
    loss_total = 0

    with torch.no_grad():
        # model.eval()
        total_predictions = []
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # LSTM is trained with batch_first=False
            x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
            y_batch = y_batch.to(device)
            
            out = model(x_batch)
            loss = criterion(out, y_batch)

            # running loss
            loss_total += loss.item() 

            _, pred = torch.max(out,1)
            total_predictions.extend(pred.detach().cpu().numpy())

            # calculate correct predictions of the batch
            correct += (pred == y_batch).sum().item()
            total_samples += len(y_batch)

        val_loss = loss_total / (i+1)
        val_acc = correct / total_samples            

    return val_loss, val_acc, total_predictions

def test_model(model_name, dataset, positional_encoding, batch_size=128):
    _, dl_test, _ = get_Dataloaders(dataset, batch_size, positional_encoding)
    
    model = torch.load(os.path.join("checkpoints", model_name))

    # model.eval()
    correct = 0
    total = 0
    
    for x_batch, y_batch in dl_test:
        x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
        y_batch = y_batch.to(device)
        
        out = model(x_batch)
        _, pred = torch.max(out,1)
        correct += (pred == y_batch).sum().item()
        total += len(y_batch)

    acc = correct / total            
    return acc


if __name__ == "__main__":

    parser = ArgumentParser()

    # model specs
    parser.add_argument('--model', type=str, default="LSTM", help='Model to be trained')
    parser.add_argument('--hidden_size', type=int, default=150, help='Number of hidden units per LSTM layer')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of sequential LSTM cells')
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], 
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
    parser.add_argument('--batch_size', type=int, default=128, help='Train and validation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='(initial) learning rate for training')

    # input manipulation
    parser.add_argument('--positional_encoding', default=False, action='store_true', help='Defines if positional encoding is added to the input features')
    parser.add_argument('--augmentation_type', type=str, default=None, help='Augmentation method used') 
    parser.add_argument('--augmentation_ratio', type=int, default=0, help='ratio of how much augmented data should be generated') 
        
    #util
    parser.add_argument('--export', default=False, action='store_true', help='Export model as .onnx')
    parser.add_argument('--confusionflow', default=False, action='store_true', help='Use confusion flow to log data')

    args = parser.parse_args()

    main()
