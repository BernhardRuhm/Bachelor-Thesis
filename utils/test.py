import os
from util import load_dataset, get_all_datasets
from datasets import DATASETS_DICT
import matplotlib.pyplot as plt
import pandas as pd

# (X_train, y_train), (X_test, y_test) = load_dataset("FordA")

# (X_train_jitter, y_train_jitter), _ = load_dataset("FordA", augmentation_type='window_warp', augmentation_ratio=2)

# types = []
# for ds in DATASETS_DICT:
#     types.append((DATASETS_DICT[ds]["type"]))

# print(len(set(types)))

# (X_train, y_train), (X_test, y_test) = load_dataset("Coffee", augmentation_ratio=5) 

datasets = get_all_datasets()
datasets_7030_split = []
datasets_500_samples = []

for ds in datasets:
    (X_train, y_train), (X_test, y_test) = load_dataset(ds)
    
    if (X_train.shape[0] < X_test.shape[0]):
        datasets_7030_split.append(ds)

    if (X_train.shape[0] > 500):
        print(ds, X_train.shape[0])
        datasets_500_samples.append(ds)

# print(datasets_7030_split)
print(datasets_500_samples)
# plt.plot(X_train[3], label='warping')
# plt.plot((X_train[3]), label='normal')
# plt.legend()
# plt.show()



weight_decay_experiments = [
    # "weight decay: 1e-4",
    # "weight decay: 5e-4",
    # "weight decay: 1e-3",
    # "data augmentation",
    # "keras BN",
    # "keras weight decay: 1e-3",
    # "keras LN",
    "keras",
    # "keras weight decay: 1e-4",
    # "keras dropout",
    "keras LN",
    # "keras LN + wd 5e-4",
    "keras LN + weight decay",
    "keras LN + Dropout",
    "keras BN",
    "keras BN + weight decay",
    "keras BN + Dropout",
    "keras BN + data augmentation",
    # "BN data aug",
    # "LN data aug",
    # "LN data aug DO",
    # "BN data aug DO",
    "pytorch LN",
    "pytorch LN + weight decay",
    "pytorch LN + Dropout",
    "pytorch + data augmentation",
]

weight_decay_experiments_path = [
    # "LSTM_PosEnc_BN:4 WD:1e-4 HS:100 NL:4 06_28_2024_00:03:59",
    # "LSTM_PosEnc_BN:4 WD:5e-4 HS:100 NL:4 06_28_2024_01:31:00",
    # "LSTM_PosEnc_BN:4 WD:1e-3 HS:100 NL:4 07_05_2024_22:45:30",
    # "LSTM_PosEnc_BN:4 DA HS:100 NL:4 07_09_2024_21:40:12",
    # "../../keras/experiments/vanilla_lstm_PosEnc BN HS:100 NL:4 07_18_2024_09:49:57",
    # "../../keras/experiments/vanilla_lstm_PosEnc WD:1e-3 HS:100 NL:4 07_16_2024_21:09:08",
    # "../../keras/experiments/vanilla_lstm_PosEnc LN HS:100 NL:4 07_17_2024_21:49:08",
    "../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_19_2024_00:50:10",
    #./../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_19_2024_22:38:19",
    #./../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_20_2024_13:56:20",
    "../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_21_2024_00:21:28",
    #./../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_22_2024_22:26:08",
    "../keras/experiments/vanilla_lstm_PosEnc HS:100 NL:4 07_23_2024_22:31:12",
    "../keras/experiments/LSTM_PosEnc_BN:4 HS:100 NL:4 07_27_2024_02:10:15",
    "../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_27_2024_21:43:14",
    "../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_29_2024_11:58:04",
    "../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_29_2024_06:31:15",
    "../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_29_2024_17:22:32",
    # "../../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_28_2024_13:33:42",
    # "../../keras/experiments/LSTM_PosEnc_BN:4 HS:100 NL:4 07_28_2024_13:54:33",
    # "../../keras/experiments/LSTM_PosEnc_BN:4 HS:100 NL:4 07_28_2024_14:06:33",
    # "../../keras/experiments/LSTM_PosEnc_BN:1 HS:100 NL:4 07_28_2024_14:34:54"
    # "LSTM_PosEnc_BN:4 HS:100 NL:4 07_28_2024_17:44:15",
    # "LSTM_PosEnc_BN:4 WD:1e-3 HS:100 NL:4 06_28_2024_02:57:57",
    # "LSTM_PosEnc_BN:4 DO:0.2 HS:100 NL:4 06_30_2024_08:34:15",
    # "LSTM_PosEnc_BN:4 DA HS:100 NL:4 07_09_2024_21:40:12"
]



datasets = ['50words', 'Cricket_X', 'FaceAll', 'FordA', 'NonInvasiveFatalECG_Thorax1', 'PhalangesOutlinesCorrect', 'UWaveGestureLibraryAll', 'wafer',
            "Two_Patterns", "SwedishLeaf", "StarLightCurves"]

experiment_results = []

for path in weight_decay_experiments_path:

    result_file = os.path.join(path, "results.csv")
    df = pd.read_csv(result_file)

    results = {} 

    for index, row in df.iterrows():
        results[row['dataset']] = row['accuracy']
    
    experiment_results.append(results)

best_results = {}

for model in weight_decay_experiments:
    best_results[model] = 0

for i in datasets:
    max_acc = 0
    name = "" 
    for exp, model in zip(experiment_results, weight_decay_experiments):

        acc = exp[i]

        if acc > max_acc:
            max_acc = acc
            name = model

    best_results[name] += 1

avg_acc = {}
for model in weight_decay_experiments:
    avg_acc[model] = 0

for exp, model in zip(experiment_results, weight_decay_experiments):
    total_acc = 0
    for i in datasets:
        total_acc += exp[i]

    total_acc /= len(datasets)
    avg_acc[model] = total_acc

print(best_results)
print(avg_acc)



