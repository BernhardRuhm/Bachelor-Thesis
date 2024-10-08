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

keras_bn1 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=1",
]

keras_bn2 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=2",
]

keras_bn3 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=3",
]

keras_bn4 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=4",
]

keras_bn5 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=5",
]

keras_bn6 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding",
    "--batch_norm=6",
]

keras_l1 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=1",
]

keras_l2 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=2",
]

keras_l3 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=3",
]

keras_l4 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=4",
]

keras_l5 = [
    "--framework=keras",
    "--hidden_size=100",
    "--n_layers=5",
]

# pytorch:  *) normal -> only missing
#           *) batchnorm -> only mising 
# keras     *) batch_norm 
#           *) data_aug

# new datasets:
    # SPECTRO: Ethanollevel
    # Devices: ElectricDevices 
    # MOTION: Haptics
    # AUDIO: Phoneme

# experiments = [lstm, lstm_postnorm, lstm_prenorm, cnn_lstm_prenorm, lstm_prenorm_customsplit]
experiments = [keras_bn1, keras_bn2, keras_bn3, keras_bn4, keras_bn5, keras_bn6, ] 

for args in experiments:
    print(args)

    if "--framework=pytorch" in args: 
        subprocess.run(["python", "pytorch/train_pytorch_models.py" ] + args)
    elif "--framework=keras" in args:
        subprocess.run(["python", "keras/train_keras_models.py" ] + args)
    else:
        print("Error: no framework specified")