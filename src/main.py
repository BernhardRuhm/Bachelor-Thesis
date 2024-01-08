import keras
from util import *
import tensorflow as tf
import numpy as np
from util import load_dataset
from focused_lstm import  FocusedLSTMLayer
from positional_encoding import PositionalEncoding
import matplotlib.pyplot as plt

if __name__ == "__main__":

     
      # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
      # train_eval_loop("lstm_fcn")
      # train_eval_loop("lstm", n_layers=1, focused=False, positional_encoding=True)

      # datasets = ["Coffee", "HandOutlines", "AtrialFibrillation", "EigenWorms"]
      # ds_dir = "xcubeai/validation_sets/"

      # for ds in datasets:
      #       _, (X_test, y_test) = load_dataset(ds)
      #       _, _, n_classes = extract_metrics(X_test, y_test)
      #       y_test = np.eye(n_classes)[y_test]

      #       np.save(os.path.join(ds_dir, "X_" + ds + ".npy"), X_test)
      #       np.save(os.path.join(ds_dir, "y_" + ds + ".npy"), y_test)

      # Datasets: Coffee: Univariate
      #           AbnormalHeartbeat: Univariate 
      #           AtrialFibrillation: Multivariate
      #           UWaveGestureLibrary: Multivariate

      n_epochs = 2
      n_hidden = 32
      batch_size = 7
      # datasets = ["Coffee", "HandOutlines", "AtrialFibrillation", "EigenWorms"]
      datasets = ["AtrialFibrillation"]

      for ds in datasets:
            vanilla_lstm = train_keras_model("vanilla_lstm", ds, 1, n_hidden, n_epochs=n_epochs, save_batch_1=True, positional_encoding=True)
            # focused_lstm = train_keras_model("focused_lstm", ds, 1, n_hidden, n_epochs=n_epochs, batch_size=batch_size, save_batch_1=True)
            # old_weights = focused_lstm.get_layer(name="rnn").get_weights()
            # layer = FocusedLSTMLayer(n_hidden)
            # layer.build((32, 286,1))
            # layer.set_weights(old_weights)
            # print(layer.get_weights())
            # vanilla_lstm_stacked = train_keras_model("vanilla_lstm", ds, 2, n_hidden, n_epochs=n_epochs, save_batch_1=True, positional_encoding=2) 
            # bidirectional_lstm = train_keras_model("bidirectional_lstm", ds, 1, 32, n_epochs=n_epochs, save_batch_1=True)
            # bidirectional_lstm_stacked = train_keras_model("bidirectional_lstm", ds, 3, 32, n_epochs=n_epochs, save_batch_1=True)

      # (X_train, y_train), (X_test, y_test) = load_dataset("Coffee")
      
      # model = tf.keras.models.load_model("./models/batch1_focused_lstm_Coffee.h5", 
      #                                    custom_objects={"FocusedLSTMLayer": FocusedLSTMLayer})
      # model.compile(run_eagerly=True)
      # model.evaluate(X_train, y_train)

      # seq_len = 100
      # vocab_size = 1000
      # embedding_dim = 1
      
      # X_train = X_train.reshape((1, 28, 286, 1))

      # model = keras.Sequential([
      #       keras.layers.InputLayer((28, 286, 1), batch_size=1),
      #       PositionalEncoding()
      # ])
            
      # pos_encoding = model(X_train)
      # print(pos_encoding.shape)
      # print(pos_encoding)
      # plt.subplots(1)
      # plt.plot(pos_encoding[0,0,:])
      # plt.plot(X_train[0,0,:])
      # plt.show()

#TODO:
      # 1: analyze focused lstm on lstm_fcn (+ maybe on lstm + conv)
      # 2: analyze inference time on positional_encoding with and without weights
      # 3: log history
      # 4: split datasets into univariant / multivariant


# batch_size = 32
# n_epochs = 500

