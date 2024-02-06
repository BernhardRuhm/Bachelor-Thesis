import tensorflow as tf
from tensorflow import keras
from keras.layers import InputLayer, LSTM, RNN, Dense 
from keras.models import Sequential 

from custom_layers import FocusedLSTMCell, PositionalEncoding 


def gen_vanilla_dense(input_dim, seq_len, n_layers, hidden_size, n_classes, batch_size, positional_encoding=False):
    model = Sequential()
    model.add(InputLayer(input_shape=(seq_len, input_dim), batch_size=batch_size))

    if positional_encoding == True:
        model.add(PositionalEncoding())

    for _ in range(n_layers-1):
        model.add(LSTM(hidden_size, return_sequences=True))

    model.add(LSTM(hidden_size))
    model.add(Dense(n_classes, activation="softmax"))

    return model

def gen_focused_dense(input_dim, seq_len, n_layers, hidden_size, n_classes, batch_size, positional_encoding=False):
    model = Sequential() 
    model.add(InputLayer(input_shape=(seq_len, input_dim), batch_size=batch_size))

    if positional_encoding == True:
        model.add(PositionalEncoding())

    for _ in range(n_layers-1):
        model.add(RNN(cell=FocusedLSTMCell(hidden_size), return_sequences=True))

    model.add(RNN(cell=FocusedLSTMCell(hidden_size)))
    model.add(Dense(n_classes, activation="softmax"))
    
    return model
    
valid_models = [
    ("vanilla_lstm", gen_vanilla_dense), 
    ("focused_lstm", gen_focused_dense)
]
