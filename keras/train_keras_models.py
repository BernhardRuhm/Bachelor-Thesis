import json
import os 
import sys
import time

import pandas as pd
import csv

from datetime import datetime

import tensorflow as tf
import keras
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.saving import load_model

from models import generate_model

from custom_layers import FocusedLSTMCell, PositionalEncoding

sys.path.append("./utils/")
from util import arg_parser, load_dataset, extract_metrics, models_dir 
from util import create_results_csv, add_results, calculate_eval_metrics, get_testing_datasets

keras.utils.set_random_seed(0)
np.random.seed(0)


class CustomCSVLogger(Callback):
    def __init__(self, filename, separator=','):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file, delimiter=self.separator)
        self.writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        self.writer.writerow([
            epoch,
            logs.get('loss'),
            logs.get('val_loss'),
            logs.get('accuracy'),
            logs.get('val_accuracy'),
            lr
        ])
        self.file.flush()
    
    def on_train_end(self, logs=None):
        self.file.close()

# class PredictionLogger(Callback):
#     def __init__(self, filename, test_data, separator=','):
#         super().__init__()
#         self.filename = filename
#         self.separator = separator
#         self.file = open(self.filename, 'w', newline='')
#         self.writer = csv.writer(self.file, delimiter=self.separator)
#         self.writer.writerow(['epoch', 'predictions'])
#         self.test_data = test_data

#     def on_epoch_end(self, epoch, logs=None):
#         pred = self.model.predict
#     def on_train_end(self, logs=None):
#         self.file.close()


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
    data_augmentation = args.data_augmentation

    # util args
    export = args.export

    # create experiment path
    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 
    path_suffix = " HS:" + str(hidden_size) + " NL:" + str(n_layers) + " " + time_stamp
    path_prefix = model_name

    if positional_encoding:
        path_prefix += "_PosEnc"  
    if batch_norm != 0:
        path_prefix += "_BN:" + str(batch_norm)

    experiment_path = os.path.join('keras/experiments', path_prefix + path_suffix)
    os.makedirs(experiment_path, exist_ok=True) 

    # Write args of experiment to a file
    with open(os.path.join(experiment_path,'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    result_file = os.path.join(experiment_path, 'results.csv')
    create_results_csv(result_file)

    datasets = get_testing_datasets()
    for ds in datasets:
        keras.backend.clear_session()
        train_time, model_file = train_model(model_name, 
                                            ds, 
                                            batch_size, 
                                            hidden_size, 
                                            n_layers, batch_norm, 
                                            dropout, 
                                            weight_decay,
                                            positional_encoding, 
                                            data_augmentation,
                                            export, 
                                            experiment_path,
                                            time_stamp, 
                                            learning_rate, 
                                            epochs) 

        acc, prec, recall = evaluate_model(model_name, model_file, ds, positional_encoding, data_augmentation, batch_size) 
        add_results(result_file, ds, acc, acc, train_time )
        


def train_model(model_name, dataset, batch_size, hidden_size, n_layers, batch_norm, dropout, weight_decay, positional_encoding, data_augmentation,
                save_batch_1, experiment_path, time_stamp, learning_rate, n_epochs): 

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset, positional_encoding, data_augmentation)
    seq_len, input_dim, n_classes = extract_metrics(X_train, y_train)

    print(X_train.shape)

    model = generate_model(model_name, batch_size, seq_len, input_dim, hidden_size, n_layers, n_classes, batch_norm, dropout)

    # TODO: modelfile to main function
    # saved as .h5 since .keras results in an error due to newer keras version (downgrade to 2.12 would resolve the issue)
    model_file = os.path.join(models_dir, model_name + "_" + dataset + "_" + time_stamp + ".h5")


    lr_scheduler = keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                       first_decay_steps=int(n_epochs / 4),
                                                       t_mul=1.0,
                                                       m_mul=1.0,
                                                       alpha=1e-4 / learning_rate)

    optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler, weight_decay=weight_decay)

    callbacks = [
        ModelCheckpoint(model_file, monitor="loss", save_best_only=True, verbose=0),
        # ReduceLROnPlateau(monitor="loss", patience=100, mode="auto", min_lr=1e-4, factor=1. / np.cbrt(2)),
        CustomCSVLogger(os.path.join(experiment_path, dataset + ".csv")),
    ]

    print("Training: %s %s" % (model_name, dataset))

    model.compile(
       optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks
    )

    train_time = time.time() - start_time

    if save_batch_1:
        model_batch1 = generate_model(model_name, 1, seq_len, input_dim, hidden_size, n_layers, n_classes, batch_norm, dropout)
        model_batch1.set_weights(model.get_weights())

        model_batch1.compile(
           optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # saved as .h5 since .keras results in an error due to newer keras version (downgrade to 2.12 would resolve the issue)
        model_batch1_file = os.path.join(models_dir, "batch_1_" + model_name + "_" + dataset + "_" + time_stamp + ".h5")
        model_batch1.save(model_batch1_file)

    return train_time, model_file

def evaluate_model(model_name, model_file, dataset, positional_encoding, data_augmentation, batch_size):
    
    _, (X_test, y_test) = load_dataset(dataset, positional_encoding, data_augmentation)

    if not os.path.exists(model_file):
        raise FileNotFoundError("Model %s does not exist" % model_file)

    model = keras.saving.load_model(model_file, compile=True) 
    
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    acc, prec, recall  = calculate_eval_metrics(y_test, y_pred)

    return acc, prec, recall 


if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args() 

    main()
