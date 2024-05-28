from util import load_dataset
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = load_dataset("FordA", normalized=True, custom_split=True)

(X_train_jitter, y_train_jitter), _ = load_dataset("FordA", normalized=True,custom_split=True, augmentation='window_warp')


plt.plot(X_train_jitter[3], label='warping')
plt.plot((X_train[3]), label='normal')
plt.legend()
plt.show()



