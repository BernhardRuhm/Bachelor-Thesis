import os 
import sys
import time

from datetime import datetime

import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.saving import load_model

from models import gen_focused_dense, gen_vanilla_dense, valid_models

from custom_layers import FocusedLSTMCell, PositionalEncoding

sys.path.append("../utils")
from util import load_dataset, extract_metrics, archiv_dir, models_dir, result_dir
from util import create_results_csv, add_results, calculate_eval_metrics, get_all_datasets, get_datasets_hiddensize 



def train_model(model_id, dataset, hidden_size, n_layers, positional_encoding, save_batch_1, 
                     n_epochs=2000, batch_size=128, learning_rate=0.001): 

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset, positional_encoding)
    seq_len, input_dim, n_classes = extract_metrics(X_train, y_train)

    model_name, gen_model = valid_models[model_id] 

    # saved as .h5 since .keras results in an error due to newer keras version (downgrade to 2.12 would resolve the issue)
    if positional_encoding:
        model_file = os.path.join(models_dir, model_name + "_posenc_" + dataset + ".h5")
    else:
        model_file = os.path.join(models_dir, model_name + "_" + dataset + ".h5")

    model = gen_model(input_dim, seq_len, n_layers, hidden_size, n_classes, batch_size, positional_encoding)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks = [
        ModelCheckpoint(model_file, monitor="loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="loss", patience=100, mode="auto", min_lr=1e-4, factor=1. / np.cbrt(2))
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
        model_batch1 = gen_model(input_dim, seq_len, n_layers, hidden_size, n_classes, 1, positional_encoding)
        model_batch1.set_weights(model.get_weights())

        model_batch1.compile(
           optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        if positional_encoding:
            model_batch1_name = "batch1_" + model_name + "_posenc"
        else:
            model_batch1_name = "batch1_" + model_name

        # saved as .h5 since .keras results in an error due to newer keras version (downgrade to 2.12 would resolve the issue)
        model_batch1_file = os.path.join(models_dir, model_batch1_name + "_" + dataset + ".h5")
        model_batch1.save(model_batch1_file)

    return train_time, model_file

def evaluate_model(model_id, dataset, positional_encoding, model_file, batch_size=128):
    
    _, (X_test, y_test) = load_dataset(dataset, positional_encoding)

    if not os.path.exists(model_file):
        raise FileNotFoundError("Model %s does not exist" % model_file)

    model = keras.saving.load_model(model_file, compile=True, 
                                    custom_objects={'FocusedLSTMCell': FocusedLSTMCell, 
                                                    'PositionalEncoding': PositionalEncoding}) 
    
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    acc, prec, recall  = calculate_eval_metrics(y_test, y_pred)

    return acc, prec, recall 


def train_eval_loop(model_id, hidden_size, n_layers, positional_encoding=False, save_batch_1=False):

    # datasets = get_all_datasets() 

    datasets = get_datasets_hiddensize(os.path.join(result_dir, "best.csv"))

    datasets = [["NonInvasiveFatalECG_Thorax1", "128"]]

    model_name, _ = valid_models[model_id] 

    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 

    if positional_encoding:
        result_file = os.path.join(result_dir, "keras_" + model_name + "_posenc" + "_results_" + time_stamp + ".csv")
    else:
        result_file = os.path.join(result_dir, "keras_" + model_name + "_results_" + time_stamp + ".csv")

    create_results_csv(result_file)

    for data in datasets:
        keras.backend.clear_session()
        train_time, model_file = train_model(model_id, data[0], int(data[1]), n_layers, positional_encoding, save_batch_1) 
        acc, prec, recall = evaluate_model(model_id, data[0], positional_encoding, model_file) 
        add_results(result_file, data[0], acc, prec, recall, train_time )

if __name__ == "__main__":

    hidden_size = [64, 128]
    n_layers = 1
    
    # train_eval_loop(0, 1, n_layers, positional_encoding=True, save_batch_1=False)
    # for hs in hidden_size:
    train_eval_loop(0, 1, n_layers, positional_encoding=False, save_batch_1=False)
    # for model_id in range(len(valid_models)):
    #     train_eval_loop(model_id, hidden_size, n_layers, positional_encoding=False, save_batch_1=True)
