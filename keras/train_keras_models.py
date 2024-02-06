import os 
import sys
import time

from datetime import datetime

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.saving import load_model

from models import gen_focused_dense, gen_vanilla_dense, valid_models

from custom_layers import FocusedLSTMCell, PositionalEncoding

sys.path.append("../utils")
from util import load_dataset, extract_metrics, archiv_dir, models_dir, result_dir 



def train_model(model_id, dataset, hidden_size, n_layers, positional_encoding, save_batch_1, 
                     n_epochs=250, batch_size=128, learning_rate=0.001): 

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
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
        ReduceLROnPlateau(monitor="loss", patience=50, mode="auto", min_lr=1e-4, factor=0.5)
    ]

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

def evaluate_model(model_id, dataset, model_file, batch_size=128):
    
    _, (X_test, y_test) = load_dataset(dataset)

    if not os.path.exists(model_file):
        raise FileNotFoundError("Model %s does not exist" % model_file)

    model = keras.saving.load_model(model_file, compile=True, 
                                    custom_objects={'FocusedLSTMCell': FocusedLSTMCell, 
                                                    'PositionalEncoding': PositionalEncoding}) 

    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)

    return results


def train_eval_loop(model_id, hidden_size, n_layers, positional_encoding=False, save_batch_1=False):

    # datasets = os.listdir(archiv_dir)  
    # datasets = datasets[:len(datasets) // 4]
    datasets = ["Coffee"]

    model_name, _ = valid_models[model_id] 

    time_stamp = datetime.now().strftime("%m_%d_%Y_%H:%M:%S") 

    if positional_encoding:
        result_file = open(os.path.join(result_dir, "keras_" + model_name + "_posenc" + "_results_" + time_stamp + ".txt"), "w")
    else:
        result_file = open(os.path.join(result_dir, "keras_" + model_name + "_results_" + time_stamp + ".txt"), "w")


    for ds in datasets:

        keras.backend.clear_session()

        print("Training: %s %s" % (model_name, ds))

        train_time, model_file = train_model(model_id, ds, hidden_size, n_layers, positional_encoding, save_batch_1) 
        loss, acc = evaluate_model(model_id, ds, model_file) 

        l = f"{ds}: loss: {loss:.4f}   accuracy: {acc:.4f}   training time: {train_time}\n"
        result_file.write(l) 
        result_file.flush()


if __name__ == "__main__":

    hidden_size = 32
    n_layers = 1

    for model_id in range(len(valid_models)):
        train_eval_loop(model_id, hidden_size, n_layers, positional_encoding=False, save_batch_1=True)
