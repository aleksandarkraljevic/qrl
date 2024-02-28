import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
from functools import reduce
from pqc import *

def plot(data_name, show, savename, smooth):
    '''
    Plots model training data.

    Parameters
    ----------
    data_name (str):
        The name of the data file, excluding the file extension.
    show (boolean):
        Whether the plot will be shown to the user.
    savename (str):
        What name the plot will be saved as. If False, then the plot is not saved.
    smooth (boolean):
        Whether savgol smoothing will be applied or not.
    '''
    data = np.load('data/'+data_name+'.npy', allow_pickle=True)
    rewards = data.item().get('rewards')
    if smooth==True:
        rewards = savgol_filter(rewards, 21, 1)
    episodes = np.arange(1, len(rewards) + 1)
    dataframe = np.vstack((rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])
    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='Episode', y='Reward')
    plt.title('Reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename+'.png')
    if show:
        plt.show()

def plot_averaged(data_names, show, savename, smooth):
    '''
    Plots an experiment's training average over all of its repetitions, including its standard errors.

    Parameters
    ----------
    data_names (list):
        A list of the data file names, excluding the file extensions.
    show (boolean):
        Whether the plot will be shown to the user.
    savename (str):
        What name the plot will be saved as. If False, then the plot is not saved.
    smooth (boolean):
        Whether savgol smoothing will be applied or not.
    '''
    n_names = len(data_names)
    data = np.load('data/'+data_names[0]+'.npy', allow_pickle=True)
    rewards = data.item().get('rewards')
    episodes = np.arange(1, len(rewards) + 1)
    for i in range(n_names-1):
        data =  np.load('data/'+data_names[i+1]+'.npy', allow_pickle=True)
        new_rewards = data.item().get('rewards')
        rewards = np.vstack((rewards, new_rewards))
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0) # standard deviation
    lower_bound = np.clip(mean_rewards-std_rewards, 0, 500)
    upper_bound = np.clip(mean_rewards+std_rewards,0, 500)
    if smooth == True:
        mean_rewards = savgol_filter(mean_rewards, 21, 1)
    dataframe = np.vstack((mean_rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])

    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='Episode', y='Reward')
    plt.fill_between(episodes, lower_bound, upper_bound, color='b', alpha=0.2)
    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/'+savename+'.png')
    if show:
        plt.show()

def compare_models(parameter_names, repetitions, show, savename, label_names, smooth):
    '''
    Plots multiple experiments' averaged training with their standard errors.

    Parameters
    ----------
    parameter_names (list):
        The list of the various experiments' parameter names, excluding "-repetition_" onwards.
    repetitions (int):
        The number of repetitions that each experiment contains.
    show (boolean):
        Whether the plot will be shown to the user.
    savename (str):
        What name the plot will be saved as. If False, then the plot is not saved.
    label_names (list):
        A list of strings representing the label name of each experiment in the plot's legend.
    smooth (boolean):
        Whether savgol smoothing will be applied or not.
    '''
    # this function requires the user to put all the experiment data in the data folder
    colors_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    plt.figure()
    sns.set_theme()

    for experiment in range(len(parameter_names)):
        data = np.load('data/'+parameter_names[experiment]+'-repetition_1.npy', allow_pickle=True)
        rewards = data.item().get('rewards')
        episodes = np.arange(1, len(rewards) + 1)
        for i in range(repetitions-1):
            data = np.load('data/'+parameter_names[experiment]+'-repetition_'+str(i+2)+'.npy', allow_pickle=True)
            new_rewards = data.item().get('rewards')
            rewards = np.vstack((rewards, new_rewards))
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)  # standard deviation
        lower_bound = np.clip(mean_rewards - std_rewards, 0, 500)
        upper_bound = np.clip(mean_rewards + std_rewards, 0, 500)
        if smooth == True:
            mean_rewards = savgol_filter(mean_rewards, 21, 1)
        dataframe = np.vstack((mean_rewards, episodes)).transpose()
        dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])

        sns.lineplot(data=dataframe, x='Episode', y='Reward', label=label_names[experiment])
        plt.fill_between(episodes, lower_bound, upper_bound, color=colors_list[experiment], alpha=0.1)

    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/' + savename + '.png')
    if show:
        plt.show()

def k_local_iterator(k, qubit_ind, possible_operations, qubits):
    if k == 0:
        yield cirq.I(qubits[qubit_ind])
        return
    for op in possible_operations:
        if qubit_ind == 0:
            yield op(qubits[qubit_ind])
        else:
            for next_op in k_local_iterator(k - 1, qubit_ind - 1, possible_operations, qubits):
                yield op(qubits[qubit_ind])*next_op
    if qubit_ind >= k:
        for next_op in k_local_iterator(k, qubit_ind - 1, possible_operations, qubits):
            yield next_op

def get_k_local(k, n_qubits):
    possible_operations = [cirq.Z, cirq.Y]
    qubits = cirq.GridQubit.rect(1, n_qubits)
    pauli_strings = []
    for j in range(1, k + 1):
        for combination in k_local_iterator(j, n_qubits - 1, possible_operations, qubits):
            pauli_strings.append(combination)
    return pauli_strings