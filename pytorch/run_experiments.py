import subprocess

lstm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--weight_decay=1e-4"
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
    "--batch_norm=4",
    "--weight_decay=1e-4"
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
experiments = [lstm, lstm_prenorm]

for args in  experiments:
    print(args)
    subprocess.run(["python", "train_pytorch_models.py" ] + args)
