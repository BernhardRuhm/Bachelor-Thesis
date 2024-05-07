import subprocess

# experiment1 = [
#     "--hidden_size=400",
#     "--n_layers=1",
#     "--positional_encoding",
# ]
# experiment2 = [
#     "--hidden_size=200",
#     "--n_layers=2",
#     "--positional_encoding",
# ]
experiment3 = [
    "--hidden_size=133",
    "--n_layers=3",
    "--positional_encoding",
]
experiment4 = [
    "--hidden_size=100",
    "--n_layers=4",
    "--positional_encoding"
]

experiments = [experiment3, experiment4]

# for i in range(3):
for args in  experiments:
        # args.append("--batch_norm="+str(i))
    print(args)
    subprocess.run(["python", "train_pytorch_models.py" ] + args)
