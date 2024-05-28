import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd 

def visualize_results(file):
    df = pd.read_csv(file)
    header = df.columns.to_list()

    datasets = df[header[0]]
    lstmfcn_results = df[header[1]]
    fcn_results = df[header[2]]
    lstm_results = df[header[3]]

    bar_width = 0.25

    # set the position of the bars on the x-axis
    r1 = range(len(datasets))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(10, 9))

    plt.bar(r1, lstmfcn_results, color='blue', width=bar_width, edgecolor='grey', label=header[1])
    plt.bar(r2, fcn_results,     color='darkorange', width=bar_width, edgecolor='grey', label=header[2])
    plt.bar(r3, lstm_results,    color='green', width=bar_width, edgecolor='grey', label=header[3])

    plt.ylim(0, 1.1)

    plt.ylabel('Accuracy')
    plt.xticks([r + bar_width for r in range(len(datasets))], datasets, rotation=30, ha='right', fontsize=8) # Rotate the tick labels by 45 degrees
    plt.title('Accuracy Comparision of ' + header[1] + ", " + header[2] + " & " + header[3])
    plt.legend(fontsize=8)

    plt.savefig('RESULTS ' + file.split('.')[0] + ".png", dpi=150, bbox_inches='tight')
    # plt.show()

def visualize_training_data(file):

    df = pd.read_csv(file)
    header = df.columns.to_list()

    s = file.split()
    model = s[0]
    dataset = s[-1].split('.')[0]

    plt.figure(1)
    plt.plot(df[header[0]], df['train loss'], df['val loss'])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(['train loss', 'val loss'], fontsize=14)
    plt.title(model + " " + dataset + " Training and Validation Loss", fontsize=16)
    plt.savefig('LOSS ' + file.split('.')[0] + ".png")

    plt.figure(2)
    plt.plot(df[header[0]], df['train acc'], df['val acc'])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(['train acc', 'val acc'], fontsize=14)
    plt.title(model + " " + dataset + " Training and Validation Accuracy", fontsize=16)
    plt.savefig('ACC ' + file.split('.')[0] + ".png")

def scatterplot_lstmfcn_accuracy(file):

    df = pd.read_csv(file)
    header = df.columns.to_list()

    markers = [".", "v","s", "*", "+", "d", "o", "^", "2", "H"]

    plt.figure(figsize=(6, 5))
    for i in range(len(markers)):
        plt.scatter(df.iloc[i, 2], df.iloc[i, 1], c='b', marker=markers[i], label='Pytorch vs Keras')
        plt.scatter(df.iloc[i, 3], df.iloc[i, 1], c='r', marker=markers[i])

    plt.plot([0, 1], [0, 1], 'k', linewidth=1)

    for i, d in enumerate(df[header[0]]):
        if d == '50words':
            plt.annotate(d, (df.iloc[i, 2] - 0.05, df.iloc[i, 1] - 0.07))

        if d == 'NonInvasiveFatalECG_Thorax1':
            plt.annotate('NonInvThor1', (df.iloc[i, 2] - 0.10, df.iloc[i, 1] - 0.07))

        if d == 'UWaveGestureLibraryAll':
            plt.annotate('UWaveAll', (df.iloc[i, 2] - 0.08, df.iloc[i, 1] - 0.07))

    plt.xticks(np.arange(0, 1.2, step=0.3), fontsize=14)
    plt.yticks(np.arange(0, 1.2, step=0.3), fontsize=14)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.xlabel('Keras Accuracy', fontsize=17)
    plt.ylabel('Pytorch Accuracy', fontsize=17)
    blue_label = mpatches.Patch(color='b', label='Pytorch vs Keras') 
    red_label = mpatches.Patch(color='r', label='Pytorch vs Original Paper Keras') 
    plt.legend(handles=[blue_label, red_label]) 
    plt.savefig('scatter.png', dpi=250)

def plot_combinded_loss(file1, file2):

    df = pd.read_csv(file1)
    df_posenc = pd.read_csv(file2)

    plt.figure(figsize=(9, 8))
    # plt.title('FCN 50words')

    # loss plot
    plt.subplot(2, 1, 1)
    plt.plot(df['train loss'], label='train loss', linewidth=1)
    plt.plot(df['val loss'], label='val loss', linewidth=1)
    plt.plot(df_posenc['train loss'] + 0.1, label='PosEnc train loss', linewidth=1)
    plt.plot(df_posenc['val loss'] + 0.1, label='PosEnc val loss', linewidth=1)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.legend(fontsize=13)
    plt.ylabel('Loss', fontsize=16)

    # lr plot
    plt.subplot(2, 1, 2)
    plt.plot(df['lr'], label='lr')
    plt.plot(df_posenc['lr'], label='PosEnc lr')

    plt.xticks(fontsize=13)
    plt.yticks(np.arange(1e-4, 1e-3 + 0.0001, step=0.0003), fontsize=13)
    # plt.ylim(1e-4, 1e-3)

    plt.legend(fontsize=13)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Learning Rate', fontsize=16)
    plt.suptitle('LSTM vs LSTM + PosEnc 50Words', fontsize=18)

    plt.savefig('combined.jpg')
# visualize_training_data("lstm 04_06_2024_14_27_40 50words.csv")
# visualize_results("results_lstmfcn.csv")

# scatterplot_lstmfcn_accuracy("../pytorch/results/LSTMFCN_vs_Paper.csv")
dir = '../pytorch/logs/'
plot_combinded_loss(dir + 'LSTM_PosEnc 04_20_2024_11:30:26 FordB.csv', dir + 'LSTM_PosEnc 04_20_2024_11:30:26 FordB.csv')

