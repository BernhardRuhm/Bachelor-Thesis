import sys
import os
import time 

from datetime import datetime
from models import valid_models

import torch
import onnx
from onnxsim import simplify

from dataloader import get_Dataloaders
sys.path.append("../utils")
from util import models_dir, result_dir, archiv_dir 

def train_model(model_id, device, dataset, hidden_size, n_layers, positional_encoding, simplify, 
                     n_epochs=250, batch_size=128, learning_rate=0.001): 
    
    dl_train, dl_test, metrics = get_Dataloaders(dataset, batch_size)
    seq_len, input_dim, n_classes = metrics

    model_name, gen_model = valid_models[model_id]
    model = gen_model(device, seq_len, input_dim, hidden_size, n_classes, n_layers, positional_encoding) 
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()

    for e in range(n_epochs):
        model.train()
        for x_batch, y_batch in dl_train:
            x_batch = torch.permute(x_batch, (1, 0, 2)).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        # eval model
        with torch.no_grad():
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
            if e % 10 == 0:
                print("Epoch: ", e, "train loss:", loss.item(), "val acc:", acc, "\n")
    
    train_time = time.time() - start_time

    # export model to .onnx
    dummy_x = torch.randn(seq_len, 1, input_dim).to(device)
    if positional_encoding:
        model_file = os.path.join(models_dir, model_name + "_posenc_" + dataset + ".onnx")
    else:
        model_file = os.path.join(models_dir, model_name + "_" + dataset + ".onnx")

    onnx_model = torch.onnx.export(model, 
                                     dummy_x, 
                                     model_file,
                                     export_params=True,
                                     input_names =  ["input"],
                                     output_names =  ["output"])

    if simplify and model_name == "vanilla_lstm":
        simplify_model(model_file)

    return loss.item(), acc, train_time


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

def train_eval_loop(model_id, hidden_size, n_layers, simplify, positional_encoding=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets = os.listdir(archiv_dir)
    # datasets = datasets[:len(datasets) // 4]
    datasets = ["Coffee"] 
    model_name, _ = valid_models[model_id] 

    if positional_encoding:
        model_name += "_posenc"

    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 
    result_file = open(os.path.join(result_dir, "pytorch_" + model_name + "_results_" + time_stamp + ".txt"), "w")

    for ds in datasets:
        print("Training: %s %s" % (model_name, ds))
        loss, acc, train_time = train_model(model_id, device, ds, hidden_size, n_layers, positional_encoding, simplify) 

        l = f"{ds}: loss: {loss:.4f}   accuracy: {acc:.4f}   training time: {train_time}\n"
        result_file.write(l) 
        result_file.flush()


if __name__ == "__main__":

    hidden_size = 32
    n_layers = 1
    
    for model_id in range(len(valid_models)):
        train_eval_loop(model_id, hidden_size, n_layers, simplify=True, positional_encoding=True)
