import subprocess

experiment1 = [
    "--hidden_size=400",
    "--n_layers=1",
    "--positional_encoding",
    "--batch_norm=3"
]
experiment2 = [
    "--hidden_size=200",
    "--n_layers=2",
    "--positional_encoding",
    "--batch_norm=3"
]
experiment3 = [
    "--hidden_size=133",
    "--n_layers=3",
    "--positional_encoding",
    "--batch_norm=4"
]
experiment4 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=5"
]

lstm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
]

lstm_postnorm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=3"
]

lstm_prenorm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4"
]

cnn_lstm_prenorm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--conv"
]

lstm_prenorm_customsplit = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--custom_split=0.3"

]

lstm_prenorm_dropout = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=5"
]

# experiments = [lstm, lstm_postnorm, lstm_prenorm, cnn_lstm_prenorm, lstm_prenorm_customsplit]
experiments = [lstm_prenorm_dropout]

for args in  experiments:
    print(args)
    subprocess.run(["python", "train_pytorch_models.py" ] + args)
