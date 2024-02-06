import numpy as np
import tensorflow as tf
import keras 
from keras.layers import Layer

class FocusedLSTMCell(Layer):
    """ Cell of a Focused LSTM
       
        Args:
            units: Output dimensionality 

        Inputs: 
            inputs: 2D Tensor of shape (batch_size, features)
            states: List of 2 Tensors with shape (batch_size, units), containing
                    the output state and cell state from the previous time step
        Outputs:
            Computions according to Focused LSTM forwarding rules.
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

        self.recurrent_bias = self.add_weight(name="recurrent_bias",
                                              shape=(self.units * 2),
                                              initializer="zeros")
        
        self.bias = self.add_weight(name="bias",
                                     shape=(self.units),
                                     initializer="zeros")

        self.built = True

    def call(self, inputs, states):
        hx, cx = states

        i_o = tf.matmul(hx, self.recurrent_kernel) + self.recurrent_bias
        i_o = tf.sigmoid(i_o)
        z = tf.matmul(inputs, self.kernel) + self.bias
        z = tf.tanh(z)
        
        i, o =  tf.split(i_o, 2, axis=1)

        cy = cx + i * z
        hy = o * tf.tanh(cy)

        return hy, [hy, cy]

class PositionalEncoding(Layer):
    """Positional Encoding layer.

    This layer calculates the position encoding as a mix of sine and cosine
    functions and adds it to its inputs. Encodings are stored as non trainable
    weights.
    Calculations of encodings from
    https://keras.io/api/keras_nlp/modeling_layers/sine_position_encoding/

    Args:
        max_wavelength: The maximum angular wavelength of the sine/cosine
            curves. 

    Inputs:
        inputs: The tensor inputs to compute an embedding for, with shape
            `(batch_size, sequence_length, input_dim)`.
    Outputs:
        Returns original inputs with added positional encodings 

    """

    def __init__(
        self,
        max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength


    def build(self, input_shape):
        seq_len = input_shape[1]
        input_dim = input_shape[2]
        self.positional_encodings = self.add_weight(name="positional_encodings",
                        shape=(seq_len, input_dim), 
                        initializer=tf.keras.initializers.Zeros(),
                        trainable=False)

        positions = tf.range(seq_len)
        positions = tf.cast(positions, self.compute_dtype)

        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)

        timescales = tf.math.pow(
            min_freq,
            tf.cast(2 * (tf.range(input_dim) // 2), self.compute_dtype)
            / tf.cast(input_dim, self.compute_dtype),
        )

        angles = tf.expand_dims(positions, 1) * tf.expand_dims(timescales, 0)

        cos_mask = tf.cast(tf.range(input_dim) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask

        self.set_weights(
            [tf.math.sin(angles) * sin_mask + tf.math.cos(angles) * cos_mask]
        )

        self.built = True

    def call(self, inputs):
        return inputs + self.positional_encodings
