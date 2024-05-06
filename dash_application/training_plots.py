import plotly.graph_objects as go
import plotly.subplots as ps
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio

# Assume you have your datasets and corresponding loss and learning rates defined somewhere

# Define your datasets
datasets = {
    'Dataset 1': {'train_loss': [1, 2, 3, 4], 'val_loss': [2, 3, 4, 5], 'learning_rate': [0.1, 0.2, 0.3, 0.4]},
    'Dataset 2': {'train_loss': [3, 4, 5, 6], 'val_loss': [4, 5, 6, 7], 'learning_rate': [0.2, 0.3, 0.4, 0.5]}
}

# Create initial figures
fig1 = ps.make_subplots(rows=1, cols=2, subplot_titles=('Train Loss', 'Validation Loss'))
fig2 = go.Figure()

# Define the Dash app
app = dash.Dash(__name__)
server = app.server

# Define layout
app.layout = html.Div([
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': dataset, 'value': dataset} for dataset in datasets.keys()],
        value=list(datasets.keys())[0]  # default value
    ),
    dcc.Graph(id='loss-graph'),
    dcc.Graph(id='learning-rate-graph')
])

# Define callback to update plots
@app.callback(
    [Output('loss-graph', 'figure'),
     Output('learning-rate-graph', 'figure')],
    [Input('dataset-dropdown', 'value')]
)
def update_plots(selected_dataset):
    # Get the selected dataset
    selected_data = datasets[selected_dataset]
    
    # Clear previous traces
    fig1.data = []
    fig2.data = []
    
    # Update subplots with selected data
    fig1.add_trace(go.Scatter(x=list(range(len(selected_data['train_loss']))), y=selected_data['train_loss'], mode='lines', name='Train Loss'), row=1, col=1)
    fig1.add_trace(go.Scatter(x=list(range(len(selected_data['val_loss']))), y=selected_data['val_loss'], mode='lines', name='Validation Loss'), row=1, col=2)
    
    fig2.add_trace(go.Scatter(x=list(range(len(selected_data['learning_rate']))), y=selected_data['learning_rate'], mode='lines', name='Learning Rate'))
    
    # Update layout
    fig1.update_layout(title='Losses Plot')
    fig2.update_layout(title='Learning Rate Plot')
    
    return fig1, fig2

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    # Export Dash app to HTML
    app.to_html("dash_app.html")
