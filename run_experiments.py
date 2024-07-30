import subprocess

# lstm = [
#     "--hidden_size=100",
#     "--n_layers=4",
#     "--positional_encoding",
#     "--weight_decay=1e-3"
# ]

# lstm_postnorm = [
#     "--hidden_size=100",
#     "--n_layers=4",
#     "--positional_encoding",
#     "--batch_norm=3"
# ]

lstm_prenorm = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
]

lstm_prenorm_augmentation= [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--data_augmentation"
]

lstm_prenorm_weightdecay1= [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--weight_decay=1e-4",
]

lstm_prenorm_weightdecay2= [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--weight_decay=5e-4",
]

lstm_prenorm_weightdecay3= [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--weight_decay=1e-3",
]

lstm_prenorm_dropout1 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--dropout=0.2"
]

lstm_prenorm_dropout2 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--dropout=0.3"
]

lstm_prenorm_dropout3 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--dropout=0.4"
]

lstm_prenorm_dropout4 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
    "--dropout=0.5"
]

keras_dataaug_dropout = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=1",
    "--dropout=0.2",
    "--data_augmentation"
]

keras_bn_dropout = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=1",
    "--dropout=0.2"
]

keras_bn_wd = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=1",
    "--weight_decay=1e-3"
]

keras_bn_da = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=1",
    "--data_augmentation"
]


pytorch_ln = [
    "--framework=pytorch",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4"
]

pytorch = [
    "--framework=pytorch",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=0"
]

# pytorch:  *) normal -> only missing
#           *) batchnorm -> only mising 
# keras     *) batch_norm 
#           *) data_aug

# experiments = [lstm, lstm_postnorm, lstm_prenorm, cnn_lstm_prenorm, lstm_prenorm_customsplit]
experiments = [keras_bn_dropout, keras_bn_wd, keras_bn_da] 

for args in experiments:
    print(args)

    if "--framework=pytorch" in args: 
        subprocess.run(["python", "pytorch/train_pytorch_models.py" ] + args)
    elif "--framework=keras" in args:
        subprocess.run(["python", "keras/train_keras_models.py" ] + args)
    else:
        print("Error: no framework specified")