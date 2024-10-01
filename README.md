# RNN-based TSC on MCU Platforms

## Installation 

Download the repository and run `pip install -r requirements.txt`.
The data has to be obtained from https://www.cs.ucr.edu/~eamonn/time_series_data/
Extract the zip into a folder called `datasets` placed in the root directoy.

## Training
Training can be started by running the `run_experiments.py` scripts. 
It supports training Keras and Pytorch LSTM networks. Network properties can be adjusted 
by providing the corresponding arguments in the script. A full description can be found
in the `arg_parser` function in `utils/util.py`
