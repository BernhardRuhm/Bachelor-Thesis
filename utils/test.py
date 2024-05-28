import numpy as np

from util import load_dataset

(X_train, y_train), (X_test, y_test) = load_dataset("ElectricDevices")

print(y_train.shape)
unique_classes = np.unique(y_train)

# Initialize a dictionary to store the count of samples for each class
class_counts = {class_label: 0 for class_label in unique_classes}

# Count the number of samples for each class
for sample in y_train:
    class_counts[sample] += 1

print(unique_classes)
print(class_counts)

print(y_test.shape)
unique_classes = np.unique(y_test)

# Initialize a dictionary to store the count of samples for each class
class_counts = {class_label: 0 for class_label in unique_classes}

# Count the number of samples for each class
for sample in y_test:
    class_counts[sample] += 1

print(unique_classes)
print(class_counts)
