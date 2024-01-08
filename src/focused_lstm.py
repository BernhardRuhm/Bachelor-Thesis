import tensorflow as tf
from keras.layers import Layer
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils

import os

class FocusedLSTMCell(Layer):
    
    def __init__(self, units, **kwargs):
        super(FocusedLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(name="kernel",
                                      shape=(input_dim, self.units),
                                      initializer="uniform")

        self.recurrent_kernel = self.add_weight(name="recurrent_kernel",
                                                shape=(self.units, self.units * 2),
                                                initializer="uniform")
        
        self.bias = self.add_weight(name="bias",
                                     shape=(self.units * 3),
                                     initializer="zeros")

        self.built = True

    def call(self, inputs, states):
        y_tm1, c_tm1 = states
      
        bi, bo, bz = tf.split(self.bias, 3)

        r = tf.matmul(y_tm1, self.recurrent_kernel)
        ri, ro = tf.split(r, 2, axis=1)
        w = tf.matmul(inputs, self.kernel)

        i = tf.sigmoid(ri + bi)
        o = tf.sigmoid(ro + bo)
        z = tf.tanh(w + bz)
        c = c_tm1 + i * z
        y = o * tf.tanh(c)

        return y, [y, c]
    

class FocusedLSTMLayer(Layer):

    def __init__(self, units, **kwargs):
        super(FocusedLSTMLayer, self).__init__(**kwargs)

        self.units = units
   
    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.seq_len = input_shape[1]
        self.batch_size = input_shape[0]


        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.input_dim, self.units),
                                      initializer="uniform")

        self.recurrent_kernel = self.add_weight(name="recurrent_kernel",
                                                shape=(self.units, self.units * 2),
                                                initializer="uniform")
        
        self.bias = self.add_weight(name="bias",
                                     shape=(self.units * 3),
                                     initializer="zeros")

        self.built = True

        self.i = 0

    def step(self, cell_inputs, cell_states):
        y_tm1 = cell_states[0]
        c_tm1 = cell_states[1]

        bi, bo, bz = tf.split(self.bias, 3)

        r = tf.matmul(y_tm1, self.recurrent_kernel)
        ri, ro = tf.split(r, 2, axis=1)
        w = tf.matmul(cell_inputs, self.kernel)

        i = tf.sigmoid(ri + bi)
        o = tf.sigmoid(ro + bo)
        z = tf.tanh(w + bz)
        c = c_tm1 + i * z
        y = o * tf.tanh(c)

        self.i += 1
        return y, [y, c]

    def call(self, inputs):

        outs = [] 

        initial_state = tf.zeros((self.batch_size, self.units))
        states = [initial_state, initial_state] 

        for i in range(self.seq_len):
            y, states = self.step(inputs[:,i,:], states)
            outs.append(y)

        return outs[-1]

# TODO:
#     1. FocusedLSTM
#     2. Adjust reshaping of Input
#     3. Bi-LSTM 
    
