import importlib, pkg_resources
importlib.reload(pkg_resources)

from helper import *

import gym
import time
from pqc import *
from collections import defaultdict
from functools import reduce
tf.get_logger().setLevel('ERROR')

class QRL():
    def __init__(self, savename, model, n_qubits, n_layers, n_actions, env_name, n_episodes, batch_size, learning_rates, gamma, state_bounds):
        '''
        Initializes the QRL hyperparameters and run settings.

        Parameters
        ----------
        savename (str):
            The name with which the model and data files will be saved.
        model (tensorflow keras model):
            The QRL model. This is either a flipped model or a data re-uploading model.
        n_qubits (int):
            The number of qubits that the PQC will use.
        n_layers (int):
            The number of layers (or depth) that the PQC will contain. The construction of one layer depends on whether the model is a flipped one or a data re-uploading one.
        n_actions (int):
            The number of actions that the agent can take in the environment.
        env_name (str):
            The name of the gym environment that is used to train the agent on.
        n_episodes (int):
            The number of total episodes that the model trains for.
        batch_size (int):
            The number of episodes that are used at each training step.
        learning_rates (list):
            A list of three learning rates that the optimizers within the PQC use in order to update the encoding, variational, and observable weights.
        gamma (float):
            The discount factor that is used in the learning algorithm.
        state_bounds (array):
            An array containing four float values that represent the bounds on the cartpole states.
        '''

        self.savename = savename
        self.model = model
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.state_bounds = state_bounds
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.env_name = env_name
        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=learning_rates[0], amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=learning_rates[1], amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rates[2], amsgrad=True)
        # Indexes of the encoding, variational and observable weights
        self.w_in, self.w_var, self.w_out = 1, 0, 2

    def gather_episodes(self):
        '''
        This function Interacts with environment in batch-wise manner.

        Returns
        -------
        trajectories (array):
            Trajectories of the gathered batch of episodes.
        '''

        trajectories = [defaultdict(list) for _ in range(self.batch_size)]
        envs = [gym.make(self.env_name) for _ in range(self.batch_size)]

        done = [False for _ in range(self.batch_size)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(self.batch_size) if not done[i]]
            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            if self.n_qubits > 4: # Encode the input state variables more than once if there are more than 4 qubits
                for qubit_ind in range(self.n_qubits-4):
                    for state_ind in range(len(normalized_states)):
                        normalized_states[state_ind] = np.append(normalized_states[state_ind], normalized_states[state_ind][qubit_ind%4])

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            # Compute policy for all unfinished envs in parallel
            states = tf.convert_to_tensor(normalized_states)
            action_probs = self.model([states])

            # Store action and transition all environments to the next state
            states = [None for i in range(self.batch_size)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(self.n_actions, p=policy)
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        return trajectories

    def compute_returns(self, rewards_history):
        '''
        This function computes discounted returns with discount factor "gamma".

        Parameters
        ----------
        reward_history (array):
            Contains the rewards that were obtained in one episode.

        Returns
        -------
        returns (list):
            Discounted returns at each time step of the corresponding episode.
        '''

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize them for faster and more stable learning
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        return returns

    @tf.function
    def reinforce_update(self, states, actions, returns):
        '''
        This function updates the weights of the PQC according to the REINFORCE learning algorithm.

        Parameters
        ----------
        states (array):
            The states of all time steps of the batch of episodes.
        actions (array):
            The actions of all time steps of the batch of episodes.
        returns (array):
            The discounted returns of all time steps of the batch of episodes.
        '''

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            logits = self.model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / self.batch_size
        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def save_data(self, rewards):
        '''
        This function saves the model's weights and training performance after its training.

        Parameters
        ----------
        rewards (list):
            A list containing the total rewards that were obtained at all episodes.
        '''
        data = {'rewards': rewards, 'n_layers': self.n_layers}
        np.save('data/' + self.savename + '.npy', data)
        self.model.save_weights('models/' + self.savename) # saves the final model weights

    def main(self):
        '''
        This function utilizes all the other functions from the QRL class to perform the process of training the model and saving the relevant data.
        '''

        # Start training the agent
        episode_reward_history = []
        print('Training progress: ' + '0/' + str(self.n_episodes))
        for batch in range(self.n_episodes // self.batch_size):
            # Gather episodes
            episodes = self.gather_episodes()

            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            # Update model parameters.
            self.reinforce_update(states, id_action_pairs, returns)

            # Store collected rewards
            for ep_rwds in rewards:
                episode_reward_history.append(np.sum(ep_rwds))

            print('Training progress: ' + str((batch+1)*self.batch_size) + '/' + str(self.n_episodes))

        if self.savename != False:
            self.save_data(episode_reward_history)


def main():
    '''
    This function initializes all the hyperparameters, creates the quantum model by utilizing pqc.py, and trains and saves the model by calling upon the QRL() class.
    '''

    env_name = "CartPole-v1"
    flipped_model = True # whether to use the flipped model or the data re-uploading model

    n_qubits = 4  # Number of qubits that the PQC of the model will consist of
    n_actions = 2  # Number of actions in the environment
    locality = 3 # The locality of the observables
    qubits = cirq.GridQubit.rect(1, n_qubits)

    if flipped_model:
        n_layers = 1  # Number of layers in the PQC
        pauli_strings = get_k_local(k=locality, n_qubits=n_qubits)
        linear_combination = [sum(pauli_strings)]
        observables = linear_combination # k-local observable: linear combination of all possible unique k-local Pauli strings, excluding Pauli-X operations
    else:
        n_layers = 5  # Number of layers in the PQC
        ops = [cirq.Z(q) for q in qubits]
        observables = [reduce((lambda x, y: x * y), ops)]  # Global observable: Z_0*Z_1*Z_2*Z_3

    n_episodes = 2000
    learning_rates = [0.1, 0.01, 0.01]
    gamma = 1
    beta = 1.0

    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    batch_size = 10

    savename = 'test'

    start = time.time()

    quantum_model = QuantumModel(qubits=qubits, n_layers=n_layers, observables=observables)

    if flipped_model:
        model = quantum_model.generate_flipped_model_policy(n_actions=n_actions, beta=beta)
    else:
        model = quantum_model.generate_model_policy(n_actions=n_actions, beta=beta)

    qrl = QRL(savename=savename, model=model, n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions,
              env_name=env_name, n_episodes=n_episodes, batch_size=batch_size, learning_rates=learning_rates,
              gamma=gamma, state_bounds=state_bounds)

    qrl.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))

if __name__ == '__main__':
    main()