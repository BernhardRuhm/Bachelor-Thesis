import plotly.graph_objects as go
import plotly.subplots as ps
import plotly.express as px
import plotly.io as pio

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import os

dir = "../pytorch/experiments"

experiments = [
# "LSTM_PosEnc HS:100 NL:4 05_06_2024_05:10:45",
# "LSTM_PosEnc HS:133 NL:3 05_06_2024_03:26:10",
# "LSTM_PosEnc_BN:1 HS:100 NL:4 05_08_2024_12:13:32", 
# "LSTM_PosEnc_BN:1 HS:133 NL:3 05_08_2024_10:18:43",
# "LSTM_PosEnc_BN:4 HS:100 NL:4 05_19_2024_03:12:25",
# "LSTM_PosEnc_BN:4 HS:133 NL:3 05_19_2024_01:26:39",
# "LSTM_PosEnc_BN:5 HS:100 NL:4 05_25_2024_17:29:42",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_25_2024_15:30:06",
# "LSTM_PosEnc_BN:5 HS:100 NL:4 05_26_2024_05:07:35",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_00:46:19",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_11:28:03",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_17:26:16 LR constant",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_18:05:58 DO 0.2",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_18:19:18 DO 0.3",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_18:35:03 DO 0.4",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_20:09:54 DO 0.6",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_20:28:35  DO 0.7",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_26_2024_23:29:26 DO 0.5",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_00:03:16",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_00:16:51",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_00:30:06",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_01:06:11",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_01:25:44",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_02:04:01",
# "LSTM_PosEnc_BN:5 HS:133 NL:3 05_27_2024_09:43:17",
# "LSTM_PosEnc_BN:4 HS:133 NL:3 05_27_2024_18:57:15",
# "LSTM_PosEnc_BN:4 HS:133 NL:3 05_27_2024_19:22:31",
# "LSTM_PosEnc_BN:4 HS:133 NL:3 05_27_2024_19:35:37",
# "LSTM_PosEnc_BN:4 HS:133 NL:3 05_27_2024_20:15:52"
"LSTM_PosEnc HS:100 NL:4 05_27_2024_22:49:18",
"LSTM_PosEnc_BN:3 HS:100 NL:4 05_27_2024_23:41:29",
"LSTM_PosEnc_BN:4 HS:100 NL:4 05_28_2024_00:40:48",
"LSTM_PosEnc_BN:4 HS:100 NL:4 05_28_2024_08:59:19",
"LSTM_PosEnc_BN:5 HS:100 NL:4 05_28_2024_10:27:39",
"LSTM_PosEnc_BN:4 HS:100 NL:4 05_28_2024_01:40:24"
]

datasets = ['50words', 'Cricket_X', 'NonInvasiveFatalECG_Thorax1', 'PhalangesOutlinesCorrect', 'FaceAll', 'wafer']

models = [
    # "hs:100  nl:4",
    # "hs:133  nl:3",
    # "hs:100  nl:4 + BN2",
    # "hs:133  nl:3 + BN2",
    # "hs:100  nl:4 + PreLN",
    # "hs:133  nl:3 + PreLN",
    # "hs:100  nl:4 + CNN",
    # "hs:133  nl:3 + CNN",
    # "hs:100  nl:4 + CNN2",
    # "hs:133  nl:3 + CNN2",
    # "hs:133  nl:3 + CNN LR",
    # "hs:133  nl:3 + CNN LR const",
    # "hs:133  nl:3 + CNN LR const DO 0.2",
    # "hs:133  nl:3 + CNN LR const DO 0.3",
    # "hs:133  nl:3 + CNN LR const DO 0.4",
    # "hs:133  nl:3 + CNN LR const DO 0.6",
    # "hs:133  nl:3 + CNN LR const DO 0.7",
    # "hs:133  nl:3 + CNN LR const DO 0.5",
    # "hs:133  nl:3 + CNN DO 0.5",
    # "hs:133  nl:3 + CNN DO 0.6",
    # "hs:133  nl:3 + CNN DO 0.6 all",
    # "hs:133  nl:3 + CNN DO 0.3 all",
    # "hs:133  nl:3 + CNN 128 DO 0.3 all",
    # "hs:133  nl:3 + CNN SGD",
    # "hs:133  nl:3 + CNN const LR DO 0.3",
    # "hs:133  nl:3 + PreLN + const LR",
    # "hs:133  nl:3 + PreLN + const LR + norm",
    # "hs:133  nl:3 + PreLN + const LR + norm + customsplit",
    # "hs:133  nl:3 + PreLN + norm + customsplit"
    "Lstm",
    "Lstm + PostNorm",
    "Lstm + PreNorm",
    "Conv + LSTM + PreNorm",
    "LSTM + PreNorm + Dropout",
    "LSTM + PreNorm + custom split"
]

# Define the Dash app
app = dash.Dash(__name__)
server = app.server

# Define layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': dataset, 'value': dataset} for dataset in datasets],
        value=datasets[0]  # default value
    ),
    dcc.Graph(id='loss-graph'),
    dcc.Graph(id='accuracy-graph')
])

# Define callback to update plots
@app.callback(
    [Output('loss-graph', 'figure'),
     Output('accuracy-graph', 'figure')],
    [Input('dataset-dropdown', 'value')]
)
def update_plots(selected_dataset):

    # get data to be plotted
    train_loss, val_loss, acc, lr = get_traindata(selected_dataset)    

    fig1 = ps.make_subplots(rows=3, cols=1, row_heights=[1., 1. ,0.5], vertical_spacing=0.1,
                        subplot_titles=('Train Loss', 'Validation Loss', "Learning Rate"))

    
    # Update subplots with selected data
    for i in range(len(train_loss.data)):
        fig1.add_trace(train_loss.data[i], row=1, col=1)
        fig1.add_trace(val_loss.data[i], row=2, col=1)

    fig1.add_trace(lr.data[0], row=3, col=1)
    # Update layout
    fig1.update_xaxes(title_text='Epochs', tickfont=dict(size=14), row=1, col=1)
    fig1.update_xaxes(title_text='Epochs', tickfont=dict(size=14), row=2, col=1)
    fig1.update_xaxes(title_text='Epochs', tickfont=dict(size=14), row=3, col=1)
    fig1.update_yaxes(title_text='Loss', tickfont=dict(size=14), row=1, col=1)
    fig1.update_yaxes(title_text='Loss', tickfont=dict(size=14), row=2, col=1)
    fig1.update_yaxes(title_text='Loss', tickfont=dict(size=14), row=3, col=1)

    fig1.update_traces(showlegend=False, row=2, col=1)
    fig1.update_traces(hoverinfo='y', hovertemplate="<br>".join([
        "loss: %{y}",
        "epoch: %{x}"
        # "trace: %{customdata}"
    ]))

    fig1.update_annotations(font_size=18)
    fig1.update_layout(height=1200, width=1600, legend=dict(font=dict(size=18)))


    fig2 = px.bar(pd.DataFrame({'model': models, 'accuracy': acc}), x='model', y='accuracy', color='accuracy')
    fig2.update_xaxes(tickfont=dict(size=15), tickangle=0)
    fig2.update_yaxes(tickfont=dict(size=15), range=[0., 1.])
    fig2.update_layout(title='Accuracies of all models', width=1500)
    
    return fig1, fig2

def get_traindata(dataset):

    df_trainloss = []
    df_valloss = []
    acc = []

    # extract train loss, val loss and accuracy of selected dataset over all experiments
    for experiment in experiments:
        dataset_file = os.path.join(dir, experiment, dataset + ".csv")

        df = pd.read_csv(dataset_file, usecols=["train loss"])
        df_trainloss.append(df)

        df = pd.read_csv(dataset_file, usecols=["val loss"])
        df_valloss.append(df)

        result_file = os.path.join(dir, experiment, "results.csv")

        df = pd.read_csv(result_file)
        acc.extend(df[df['dataset'] == dataset]['accuracy'].values)


    df_trainloss = pd.concat(df_trainloss, axis=1, ignore_index=True)
    df_trainloss.columns = models

    df_valloss= pd.concat(df_valloss, axis=1, ignore_index=True)
    df_valloss.columns = models

    # extract lr over epochs
    df_lr = pd.read_csv(dataset_file, usecols=["lr"])

    # convert to plotable data
    train_loss = px.line(df_trainloss, x=df_trainloss.index, y=df_trainloss.columns)
    val_loss = px.line(df_valloss, x=df_valloss.index, y=df_valloss.columns)
    lr = px.line(df_lr, x=df_lr.index, y=df_lr.columns)

    return train_loss, val_loss, acc, lr

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
