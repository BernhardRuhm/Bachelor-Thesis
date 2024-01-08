import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import InputLayer, LSTM, GRU,Dense, RNN, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Input, Dropout, concatenate, Bidirectional
from keras.models import Sequential, Model 

from custom_layers import FocusedLSTMCell, PositionalEncoding 


def gen_vanilla_dense(input_dim, seq_len, n_layers, n_hidden, n_classes, batch_size, positional_encoding=False):
    model = Sequential()
    model.add(InputLayer(input_shape=(seq_len, input_dim), batch_size=batch_size))

    if positional_encoding == True:
        model.add(PositionalEncoding())

    for _ in range(n_layers-1):
        model.add(LSTM(n_hidden, return_sequences=True))

    model.add(LSTM(n_hidden))
    model.add(Dense(n_classes, activation="softmax"))

    return model

def gen_focused_dense(input_dim, seq_len, n_layers, n_hidden, n_classes, batch_size, positional_encoding=False):
    model = Sequential() 
    model.add(InputLayer(input_shape=(seq_len, input_dim), batch_size=batch_size))

    if positional_encoding == True:
        model.add(PositionalEncoding())

    for _ in range(n_layers-1):
        model.add(RNN(cell=FocusedLSTMCell(n_hidden), return_sequences=True))

    model.add(RNN(cell=FocusedLSTMCell(n_hidden)))
    model.add(Dense(n_classes, activation="softmax"))
    
    return model


def LSTMFCN(input_dim, seq_len, n_classes, n_hidden, focused):

    ip = Input(shape=(seq_len, 1))

    if focused:
        x = RNN(FocusedLSTMCell(n_hidden))(ip)
    else:
        x = LSTM(n_hidden)(ip)
        # x = Dropout(0.8)(x)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(ip, out)

    return model

def LSTMConv(input_dim, seq_len, n_layers, n_hidden, n_classes, batch_size, focused):
    
    model = Sequential()
    model.add(InputLayer(input_shape=(seq_len, input_dim)))

    model.add(LSTM(n_hidden, return_sequences=True))
    model.add(LSTM(n_hidden, return_sequences=True))

    model.add(Conv1D(64, 5, kernel_initializer="he_uniform"))
    model.add(ReLU())


    model.add(Dense(n_classes, activation="softmax"))

    return model

valid_models = [
    ("vanilla_lstm", gen_vanilla_dense), 
    ("focused_lstm", gen_focused_dense)
]
