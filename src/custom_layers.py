import numpy as np
import tensorflow as tf
import keras 
from keras.layers import Layer

class FocusedLSTMCell(Layer):
    """ Cell of a Focused LSTM
        
        Inputs: 
            inputs: 2D Tensor of shape (batch_size, features)
            states: List of 2 Tensors with shape (batch_size, units), containing
                    the output state and cell state from the previous time step
        Outputs:
    """
    
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
    

class PositionalEncoding(Layer):
    """ Positional Encoding Layer for Time Series Data

        Inputs: 
            3D Tensor of shape (batch_size, timesteps, features)

        Output:
            3D Tensor if shape (batch_size, timesteps, features) with added positional encodings
    """

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seq_len = input_shape[-2]
        self.built = True

    def call(self, inputs):

        positions = np.sin(np.arange(self.seq_len) + 1)
        positions = np.reshape(positions, (1, self.seq_len, 1))
        
        return inputs + tf.convert_to_tensor(positions, dtype=tf.float32)
