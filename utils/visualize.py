import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os

def visualize_results(path, file):
    df = pd.read_csv(os.path.join(path, file))
    header = df.columns.to_list()

    datasets = df[header[0]]
    lstmfcn_results = df[header[1]]
    fcn_results = df[header[2]]
    # lstm_results = df[header[3]]

    bar_width = 0.25

    # set the position of the bars on the x-axis
    r1 = range(len(datasets))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(10, 9))

    plt.bar(r1, lstmfcn_results, color='blue', width=bar_width, edgecolor='grey', label=header[1])
    plt.bar(r2, fcn_results,     color='darkorange', width=bar_width, edgecolor='grey', label=header[2])
    # plt.bar(r3, lstm_results,    color='green', width=bar_width, edgecolor='grey', label=header[3])

    plt.ylim(0, 1.1)

    plt.ylabel('Accuracy')
    plt.xticks([r + bar_width for r in range(len(datasets))], datasets, rotation=30, ha='right', fontsize=8) # Rotate the tick labels by 45 degrees
    # plt.title('Accuracy Comparision of ' + header[1] + ", " + header[2] + " & " + header[3])
    plt.title('Accuracy Comparision of ' + header[1] + " & " + header[2])
    plt.legend(fontsize=8)

    plt.savefig(path + file.split('.')[0] + ".png", dpi=150, bbox_inches='tight')
    plt.close
    # plt.show()

def visualize_training_data(file, path):

    df = pd.read_csv(os.path.join(dir, file))
    model = file.split('.')[0]
    print(file)
    modelname = model.split()[0]
    dataset = model.split()[-1]

    plt.figure(figsize=(9, 5))
    df['train loss'].plot()
    df['val loss'].plot()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(['train loss', 'val loss'], fontsize=14)
    plt.title(modelname + " " + dataset + " Training and Validation Loss", fontsize=16)
    plt.savefig('../pytorch/plots/LOSS ' + model + ".png")
    plt.close()

# dir = "../pytorch/logs"
# for file in os.listdir(dir):
#     visualize_training_data(file, "../pytorch/plots")

visualize_results("../pytorch/results/", "LSTM_PosEnc.csv")