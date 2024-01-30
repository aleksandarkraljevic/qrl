import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque
from scipy.signal import savgol_filter
from pqc import *

def exponential_anneal(t, start, final, decay_constant):
    '''
    Exponential annealing scheduler for epsilon-greedy policy.

    Parameters
    ----------
    t (int):
        Current timestep.
    start (float):
        Initial epsilon value.
    final (float):
        Lowest possible value that epsilon can get to.
    decay_constant (float):
        The speed at which epsilon decays.
    '''
    return final + (start - final) * np.exp(-decay_constant*t)


def boltzmann_exploration(actions, temperature):
    '''
    Boltzmann exploration policy.

    Parameters
    ----------
    actions (list):
        Vector with possible actions.
    temperature (float):
        Strength of the exploration.
    '''
    actions = actions[0] - np.max(actions[0])
    a = actions / temperature  # scale by temperature
    return np.exp(a)/np.sum(np.exp(a))

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
    n_holes = data.item().get('n_holes')
    memory_size = 2 * (n_holes - 2)
    if smooth==True:
        rewards = savgol_filter(rewards, 71, 1)
    episodes = np.arange(1, len(rewards) + 1)
    dataframe = np.vstack((rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])
    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='Episode', y='Reward')
    plt.ylim(-1 * memory_size, 0)
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
    n_holes = data.item().get('n_holes')
    memory_size = 2 * (n_holes - 2)
    rewards = data.item().get('rewards')
    episodes = np.arange(1, len(rewards) + 1)
    for i in range(n_names-1):
        data =  np.load('data/'+data_names[i+1]+'.npy', allow_pickle=True)
        new_rewards = data.item().get('rewards')
        rewards = np.vstack((rewards, new_rewards))
    mean_rewards = np.mean(rewards, axis=0)
    se_rewards = np.std(rewards, axis=0) / np.sqrt(n_names) # standard error
    lower_bound = np.clip(mean_rewards-se_rewards, None , 0)
    upper_bound = np.clip(mean_rewards+se_rewards, None, 0)
    if smooth == True:
        mean_rewards = savgol_filter(mean_rewards, 71, 1)
    dataframe = np.vstack((mean_rewards, episodes)).transpose()
    dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])

    plt.figure()
    sns.set_theme()
    sns.lineplot(data=dataframe, x='Episode', y='Reward')
    plt.fill_between(episodes, lower_bound, upper_bound, color='b', alpha=0.2)
    plt.ylim(-1*memory_size,0)
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
        n_holes = data.item().get('n_holes')
        memory_size = 2*(n_holes-2)
        rewards = data.item().get('rewards')
        episodes = np.arange(1, len(rewards) + 1)
        for i in range(repetitions-1):
            data = np.load('data/'+parameter_names[experiment]+'-repetition_'+str(i+2)+'.npy', allow_pickle=True)
            new_rewards = data.item().get('rewards')
            rewards = np.vstack((rewards, new_rewards))
        mean_rewards = np.mean(rewards, axis=0)
        se_rewards = np.std(rewards, axis=0) / np.sqrt(repetitions)  # standard error
        lower_bound = np.clip(mean_rewards - se_rewards, None, 0)
        upper_bound = np.clip(mean_rewards + se_rewards, None, 0)
        if smooth == True:
            mean_rewards = savgol_filter(mean_rewards, 71, 1)
        dataframe = np.vstack((mean_rewards, episodes)).transpose()
        dataframe = pd.DataFrame(data=dataframe, columns=['Reward', 'Episode'])

        sns.lineplot(data=dataframe, x='Episode', y='Reward', label=label_names[experiment])
        plt.fill_between(episodes, lower_bound, upper_bound, color=colors_list[experiment], alpha=0.1)
        plt.ylim(-1 * memory_size, 0)

    plt.title('Mean reward per episode')
    if savename != False:
        plt.savefig('plots/' + savename + '.png')
    if show:
        plt.show()

def evaluate(model_name, n_samples, print_strategy, print_evaluation, plot_distribution, save):
    '''
    Evaluates a single model by looking at its best-found strategy, and averaging its performance. It also plots the distribution of all the numbers of needed guesses after averaging.

    Parameters
    ----------
    model_name (str):
        The name of the model file, excluding the file extension.
    n_samples (int):
        The amount of samples that the model will be averaged over.
    print_strategy (boolean):
        Whether the strategy of the model will be printed.
    print_evaluation (boolean):
        Whether the performance of the model will be printed.
    plot_distribution (boolean):
        Whether the distribution of the numbers of needed guesses of all samples will be plotted.
    save (boolean):
        Whether the plots will be saved using the same name as parameter_name.
    '''
    data = np.load('data/'+model_name+'.npy', allow_pickle=True)
    n_holes = data.item().get('n_holes')
    n_layers = data.item().get('n_layers')
    memory_size = 2 * (n_holes - 2)
    qubits = cirq.GridQubit.rect(1, memory_size)
    ops = [cirq.Z(q) for q in qubits]
    observables = []
    for i in range(n_holes):
        observables.append(ops[i])
    quantum_model = QuantumModel(qubits, n_layers, observables)
    model = quantum_model.generate_model_Qlearning(False)
    model.load_weights('models/' + model_name)
    env = FoxInAHole(n_holes)
    episode_lengths = []
    episode_rewards = []
    if print_strategy:
        done = env.reset()
        state = deque([0]*memory_size, maxlen=memory_size)
        for step in range(memory_size):
            state_tensor = tf.convert_to_tensor([np.array(state)])
            q_vals = model([state_tensor])
            action = np.argmax(q_vals) + 1
            state.append(action)
        print("The strategy for the first 2(n-2) guesses =", list(state))
    for sample in range(n_samples):
        current_episode_length = 0
        episode_reward = 0
        done = env.reset()
        state = deque([0]*memory_size, maxlen=memory_size)
        while not done:
            current_episode_length += 1
            state_tensor = tf.convert_to_tensor([np.array(state)])
            q_vals = model([state_tensor])
            action = np.argmax(q_vals)
            reward, done = env.guess(action)
            episode_reward += reward
            new_observation = state.copy()
            new_observation.append(action)
            state = new_observation
            env.step()
        episode_lengths.append(current_episode_length)
        episode_rewards.append(episode_reward)

    if print_evaluation:
        print('The average amount of guesses needed to finish the game is:', round(np.mean(episode_lengths), 2), '+-',
              round(np.std(episode_lengths) / np.sqrt(n_samples), 2))
        print('The average reward per game is:', round(np.mean(episode_rewards), 2), '+-',
              round(np.std(episode_rewards) / np.sqrt(n_samples), 2))

    if plot_distribution:
        episode_rewards = [x * (-1) + 1 for x in episode_rewards]
        episode_rewards = pd.DataFrame(episode_rewards, columns=["# guesses"])
        plt.figure()
        sns.histplot(episode_rewards, x="# guesses")
        plt.title("Distribution of the guess count")
        if save:
            plt.savefig('plots/' + model_name + '-distribution.png')
        plt.show()

    return np.mean(episode_lengths)