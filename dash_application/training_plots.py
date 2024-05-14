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
"LSTM_PosEnc HS:100 NL:4 05_06_2024_05:10:45",
"LSTM_PosEnc HS:133 NL:3 05_06_2024_03:26:10",
# "LSTM_PosEnc HS:200 NL:2 05_06_2024_02:50:00",
# "LSTM_PosEnc HS:400 NL:1 05_06_2024_02:16:42",
"LSTM_PosEnc_BN:1 HS:100 NL:4 05_06_2024_08:58:06",
"LSTM_PosEnc_BN:1 HS:133 NL:3 05_06_2024_07:09:10",
# "LSTM_PosEnc_BN:1 HS:200 NL:2 05_06_2024_06:25:38",
# "LSTM_PosEnc_BN:1 HS:400 NL:1 05_06_2024_05:45:38",
"LSTM_PosEnc_BN:1 HS:100 NL:4 05_08_2024_12:13:32", 
"LSTM_PosEnc_BN:1 HS:133 NL:3 05_08_2024_10:18:43",
# "LSTM_PosEnc_BN:1 HS:100 NL:4 05_09_2024_13:33:43",
# "LSTM_PosEnc_BN:1 HS:133 NL:3 05_09_2024_11:42:34",
# "LSTM_PosEnc_BN:2 HS:100 NL:4 05_06_2024_12:57:08",
# "LSTM_PosEnc_BN:2 HS:133 NL:3 05_06_2024_10:58:32",
# "LSTM_PosEnc_BN:2 HS:200 NL:2 05_06_2024_10:15:35",
# "LSTM_PosEnc_BN:2 HS:400 NL:1 05_06_2024_09:37:34"
"LSTM_PosEnc_BN:3 HS:100 NL:4 05_14_2024_00:43:36",
"LSTM_PosEnc_BN:3 HS:133 NL:3 05_13_2024_22:48:45"
]

datasets = ['50words' ,'Cricket_X', 'NonInvasiveFatalECG_Thorax1'] 
models = [
    "hs:100  nl:4",
    "hs:133  nl:3",
    "hs:100  nl:4 + BN",
    "hs:133  nl:3 + BN",
    "hs:100  nl:4 + BN2",
    "hs:133  nl:3 + BN2",
    "hs:100  nl:4 + LN",
    "hs:133  nl:3 + LN",
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
