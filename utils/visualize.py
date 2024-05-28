import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os

def visualize_results(path, file):
    df = pd.read_csv(os.path.join(path, file))
    header = df.columns.to_list()

    datasets = df[header[0]]
    fcn_results = df[header[1]]
    fcn_results_pos = df[header[2]]
    lstm_results = df[header[3]]
    best_lstm_results = df[header[4]]

    bar_width = 0.15

    # set the position of the bars on the x-axis
    r1 = range(len(datasets))
    r2 = [x + bar_width+0.05 for x in r1]
    r3 = [x + bar_width+0.05 for x in r2]
    r4 = [x + bar_width+0.05 for x in r3]

    plt.figure(figsize=(12, 9))

    plt.bar(r1, fcn_results,     color='blue', width=bar_width, edgecolor='grey', label=header[1])
    plt.bar(r2, fcn_results_pos, color='red', width=bar_width, edgecolor='grey', label=header[2])
    plt.bar(r3, lstm_results,    color='gold', width=bar_width, edgecolor='grey', label=header[3])
    plt.bar(r4, best_lstm_results, color='green', width=bar_width, edgecolor='grey', label=header[4])

    plt.ylim(0, 1.1)

    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks([r + 2* bar_width for r in range(len(datasets))], datasets, rotation=30, ha='right', fontsize=10) # Rotate the tick labels by 45 degrees
    plt.yticks(fontsize=12)
    # plt.title('Accuracy of ' + header[1] + ", " + header[2] + " & " + header[3], fontsize=16)
    # plt.title('Accuracy of ' + header[1] + " & " + header[2] + " & " + header[3], fontsize=16)
    plt.title('Accuracy Comparision FCN & LSTM', fontsize=16)
    plt.legend(fontsize=12)

    # plt.savefig(path + file.split('.')[0] + ".png", dpi=150, bbox_inches='tight')
    plt.savefig("LSTM.png", dpi=150, bbox_inches='tight')
    plt.close
    # plt.show()

def visualize_training_data(file, path):

    df = pd.read_csv(os.path.join(dir, file))
    model = file.split('.')[0]
    print(file)
    modelname = model.split()[0]
    dataset = model.split()[-1]

    plt.figure(figsize=(9, 8))
    plt.subplot(2, 1, 1)

    plt.plot(df['train loss'], label='train loss', linewidth=1)
    plt.plot(df['val loss'], label='val loss', linewidth=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.legend(fontsize=13)
    plt.ylabel('Loss', fontsize=16)

    plt.subplot(2, 1, 2)
    plt.plot(df['lr'][:-2], label='lr')

    plt.xticks(fontsize=13)
    plt.yticks(np.arange(1e-4, 1e-3 + 0.0001, step=0.0003), fontsize=13)

    plt.legend(fontsize=14)
    plt.suptitle(modelname + " " + dataset + " Loss + lr", fontsize=16)
    plt.savefig('../pytorch/trainingdata_plots/LOSS ' + model + ".png")
    plt.close()

# dir = "../pytorch/logs"
# for file in os.listdir(dir):
# visualize_training_data("LSTM_PosEnc 04_21_2024_23:53:25 UWaveGestureLibraryAll.csv", "../pytorch/trainingdata_plots/") 
# visualize_results(".", "./critical_distance_all.csv")
