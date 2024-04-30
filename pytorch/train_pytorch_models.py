import sys
import os
import time 
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import onnx
from onnxsim import simplify

from dataloader import get_Dataloaders
from models import valid_models, LSTMFCN, init_weights

sys.path.append("../utils")
from util import models_dir, create_results_csv, add_results 
from visualize import visualize_training_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)

# datasets = ['50words', 'ChlorineConcentration', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
# 'ElectricDevices', 'FordA', 'FordB', 'NonInvasiveFatalECG_Thorax1', 'UWaveGestureLibraryAll'] 
datasets = ['Cricket_X']

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
        print("input shape", x_batch.shape)

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

# TODO: save predictions (in log file or with confusion flow)
def evaluate(model, criterion, dataloader):
    correct = 0
    total_samples = 0
    loss_total = 0

    with torch.no_grad():
        model.eval()
        for i, (x_batch, y_batch) in enumerate(dataloader):
            # LSTM is trained with batch_first=False
            x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
            y_batch = y_batch.to(device)
            
            out = model(x_batch)
            loss = criterion(out, y_batch)

            # running loss
            loss_total += loss.item() 

            # calculate correct predictions of the batch
            _, pred = torch.max(out,1)
            correct += (pred == y_batch).sum().item()
            total_samples += len(y_batch)

        val_loss = loss_total / (i+1)
        val_acc = correct / total_samples            
    
    return val_loss, val_acc

def train_model(model_id, model_name, dataset, hidden_size, n_layers, filters, positional_encoding, simplify, 
                     n_epochs=2000, batch_size=128, learning_rate=0.001): 
    
    train_data = [] 
    kernels = [3, 5, 8]

    dl_train, dl_test, metrics = get_Dataloaders(dataset, batch_size, positional_encoding)
    seq_len, input_dim, n_classes = metrics

    _, gen_model = valid_models[model_id]
    checkpoint = os.path.join("checkpoints", model_name)

    model = gen_model(device, input_dim, hidden_size, n_classes, n_layers, filters, kernels)
    model.apply(init_weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(n_epochs/4), eta_min=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=1. / np.cbrt(2),
    #     patience=100,
    #     min_lr=1e-4
    # )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    old_lr = learning_rate
    new_lr = learning_rate
    start_time = time.time()

    for e in range(n_epochs):
        model.train()
        correct = 0
        total_samples = 0
        total_train_loss = 0

        for i, (x_batch, y_batch) in enumerate(dl_train):
            x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            out = model(x_batch)
            loss = criterion(out, y_batch)
            total_train_loss += loss.item()

            _, pred = torch.max(out,1)
            correct += (pred == y_batch).sum().item()
            total_samples += len(y_batch)

            loss.backward()
            optimizer.step()

        scheduler.step()
        # reinitialze optimizer when lr is updated
        # new_lr = get_lr(optimizer)
        # if (old_lr != new_lr):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
        # old_lr = new_lr

        train_loss_per_epoch = total_train_loss / (i+1)
        train_acc = correct / total_samples

        # eval model
        correct = 0
        total_samples = 0
        total_val_loss = 0

        with torch.no_grad():
            model.eval()
            
            for i, (x_batch, y_batch) in enumerate(dl_test):
                x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
                y_batch = y_batch.to(device)
                
                out = model(x_batch)
                loss = criterion(out, y_batch)
                total_val_loss += loss.item() 

                _, pred = torch.max(out,1)
                correct += (pred == y_batch).sum().item()
                total_samples += len(y_batch)

            val_loss_per_epoch = total_val_loss / (i+1)
            val_acc = correct / total_samples            

            if e % 10 == 0:
                print("Epoch:", e, "train_loss:", loss.item(), "val_acc:", val_acc, "lr:", get_lr(optimizer), "\n")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc 
                torch.save(model, checkpoint)
        
        train_data.append({'train loss': train_loss_per_epoch, 'val loss': val_loss_per_epoch,
                           'train acc': train_acc, 'val acc': val_acc, 'lr': get_lr(optimizer)})

    df = pd.DataFrame(train_data)
    # visualize_training_data(df, model_name, dataset, n_epochs)
    df.to_csv(os.path.join("logs", model_name + " " + dataset+ ".csv"))

    train_time = time.time() - start_time
    
    export_model(checkpoint, model_name, dataset, seq_len, input_dim, device)

    return train_time

def test_model(model_name, dataset, positional_encoding, batch_size=128):
    _, dl_test, _ = get_Dataloaders(dataset, batch_size, positional_encoding)
    
    model = torch.load(os.path.join("checkpoints", model_name))

    model.eval()
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

def train_eval_loop(model_id, hidden_size, n_layers, filters, simplify, positional_encoding=False):

    model_name, _ = valid_models[model_id] 

    if positional_encoding:
        model_name += "_PosEnc"

    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 
    model_name += " " + time_stamp
    result_file = os.path.join("results", model_name +  " filters:" + " ".join(map(str, filters)) + 
                               " hidden:" + str(hidden_size) + " nlayers:" + str(n_layers)+ ".csv")

    create_results_csv(result_file)

    for ds in sorted(datasets):
        print("Training: %s %s" % (model_name, ds))
        train_time = train_model(model_id, model_name, ds, hidden_size, n_layers, filters, positional_encoding, simplify) 
        acc = test_model(model_name, ds, positional_encoding)
        add_results(result_file, ds, acc, train_time)


if __name__ == "__main__":

    hidden_size = [200, 100, 67, 50]
    n_layers = [1, 2, 3, 4]    
    filters = [128, 256, 128]

    # for hs in hidden_size:
        # for nl in n_layers:

    # train_eval_loop(1, 200, 4, filters, simplify=False, positional_encoding=True)
    train_eval_loop(1, 200, 1, filters, simplify=False, positional_encoding=False)
    train_eval_loop(1, 100, 2, filters, simplify=False, positional_encoding=False)
    train_eval_loop(1, 67,  3, filters, simplify=False, positional_encoding=False)
    train_eval_loop(1, 50,  4, filters, simplify=False, positional_encoding=False)


    # FCN
    # train_eval_loop(2, hidden_size, n_layers, filters, simplify=False, positional_encoding=False)
    # train_eval_loop(2, hidden_size, n_layers, filters, simplify=False, positional_encoding=True)
