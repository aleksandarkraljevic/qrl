# Flipped models and classical shadows of quantum reinforcement learning, by Aleksandar Kraljevic

The code consists of eight .py files:
- quantum_model.py:
    This is the file that runs the training of either a flipped or data re-uploading quantum RL model. It also saves the model's final weights and its performance throughout the training procedure.
- pqc.py:
    This file contains the code which creates a flipped or data re-uploading model for quantum_model.py.
- evaluate_model.py:
    This code is for evaluating experiments that have already been performed.
- experiment.py:
    This code is able to run singular experiments for chosen hyperparameters using quantum_model.py and pqc.py.
- tuning_experiment.py:
    This code is able to run batches of experiments using quantum_model.py and pqc.py. It was used to hyperparameter tune flipped models in a batch-wise manner via ALICE, which is a computational cluster of Leiden University.
- gtp.py:
    This is the file that runs the training of a GTP model which represents the aforementioned flipped models with "infinite" circuit depth. It also saves the model's performance throughout the training procedure.
- gtp_tuning_experiment.py:
    This code is able to run batches of experiments using gtp.py. It was used to hyperparameter tune GTP models in a batch-wise manner via ALICE, which is a computational cluster of Leiden University.
- helper.py:
    This file contains various functions that are used by the other python files, in order to not clutter them. Its job is to plot data, compare results and compute k-local Pauli strings.

## Dependencies
The code uses some python packages that need to be installed to run:
- tensorflow==2.15.0
- tensorflow-quantum==0.7.3
- gym==0.18.0
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- cirq
- sympy
- argparse
- collections
- functools
- time
- importlib
- pkg_resources
- itertools

## Running the code
To run any of the python files, make sure to have all the .py files in the same folder. In addition to this, make sure to create three empty folders called "data", "models", and "plots". As the names suggest, these are the files that the data, models, and plots will be saved in.
