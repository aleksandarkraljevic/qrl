import importlib, pkg_resources
importlib.reload(pkg_resources)

from helper import *

import gym, cirq, sympy
import time
import numpy as np
import itertools
from collections import deque, defaultdict
tf.get_logger().setLevel('ERROR')

class QRL():
    def __init__(self, savename, locality, n_qubits, n_actions, env_name, n_episodes, batch_size, learning_rates, gamma, state_bounds, breakout):
        '''
        Initializes the QRL parameters.

        Parameters
        ----------
        savename (str):
            The name with which the file model and data files will be saved.
        n_qubits (int):
            The number of qubits that the PQC will use.
        n_actions (int):
            The number of actions that the agent can take in the environment.
        env_name (str):
            The name of the gym environment that is used to train the agent on.
        n_episodes (int):
            The amount of total episodes that the model trains for.
        batch_size (int):
            The amount of samples that each training batch consists of.
        learning_rates (list):
            A list of three learning rates that the optimizers within the PQC use in order to update the encoding, variational, and rescaling weights.
        gamma (float):
            The discount factor that is used in the Q-learning algorithm.
        state_bounds (array):
            An array containing four float values that represent the bounds on the cartpole states.
        breakout (boolean):
            A boolean value that decides whether the agent should stop its training after it has "beaten" the game.
        '''
        self.savename = savename
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.state_bounds = state_bounds
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.env_name = env_name
        self.breakout = breakout
        self.locality = locality
        self.optimizer_coeff = tf.keras.optimizers.Adam(learning_rate=learning_rates[0], amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=learning_rates[1], amsgrad=True)
        # Indexes of the weights for each of the parts of the circuit
        self.coeff, self.w_out = 0, 1

    def gather_episodes(self):
        """Interact with environment in batched fashion."""

        trajectories = [defaultdict(list) for _ in range(self.batch_size)]
        envs = [gym.make(self.env_name) for _ in range(self.batch_size)]

        done = [False for _ in range(self.batch_size)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(self.batch_size) if not done[i]]
            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            if self.n_qubits > 4:
                for qubit_ind in range(self.n_qubits-4):
                    for state_ind in range(len(normalized_states)):
                        normalized_states[state_ind] = np.append(normalized_states[state_ind], normalized_states[state_ind][qubit_ind%4])

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            # Compute policy for all unfinished envs

            action_probs = []

            for state in normalized_states:
                action_prob = self.GTP(state, self.coeff.numpy()[0][0], self.coeff.numpy()[0][1:len(self.omegas)+1], self.coeff.numpy()[0][len(self.omegas)+1:])
                action_prob *= self.w.numpy()
                action_probs.append(action_prob)

            # Store action and transition all environments to the next state
            states = [None for i in range(self.batch_size)]
            for i, policy in zip(unfinished_ids, action_probs):
                action = np.random.choice(self.n_actions, p=policy)
                print(action)
                exit()
                states[i], reward, done[i], _ = envs[i].step(action)
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)

        return trajectories

    def compute_returns(self, rewards_history):
        """Compute discounted returns with discount factor `gamma`."""
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

    def GTP(self, state, c_zero, a_w, b_w):
        # states is the input, omegas are the frequency combinations, c_0, a_w and b_w are the trainable variables.
        # c_0 is the coefficient for the all 0 frequency, a_w is the real part of the c_w coefficient, b_w is the imaginary part of the c_w coefficient
        # this function only works correctly if each qubit only has 1 Pauli rotation as its encoding gate

        all_elements = []
        all_elements.append(c_zero)
        for i in range(len(self.omegas) - 1):
            element = 2 * a_w[i] * np.cos(np.dot(self.omegas[i], state)) - 2 * b_w[i] * np.sin(np.dot(self.omegas[i], state))
            all_elements.append(element)
        return np.sum(all_elements)

    @tf.function
    def reinforce_update(self, states, actions, returns):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        returns = tf.convert_to_tensor(returns)

        with tf.GradientTape() as tape:
            print([self.w, self.coeff])
            exit()
            tape.watch([self.w, self.coeff])
            logits = self.model(states)
            p_actions = tf.gather_nd(logits, actions)
            log_probs = tf.math.log(p_actions)
            loss = tf.math.reduce_sum(-log_probs * returns) / self.batch_size
        grads = tape.gradient(loss, self.model.trainable_variables)
        for optimizer, w in zip([self.optimizer_coeff, self.optimizer_out], [self.w_coeff, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model.trainable_variables[w])])

    def save_data(self, rewards):
        '''
        Saves the model after its training, as well as important results and properties.

        Parameters
        ----------
        rewards (list):
            A list of all the total rewards that were obtained at the end of each episode.
        '''
        data = {'rewards': rewards}
        np.save('data/' + self.savename + '.npy', data)

    def main(self):

        omegas = itertools.product([-1, 0, 1], repeat=self.n_qubits)
        omegas = list(omegas)
        self.omegas = omegas[:int(np.floor((len(omegas) / 2)))]  # get all the frequencies that are unique with regards to the combination, regardless of vector sign, and not counting the all zero

        pauli_strings = get_k_local(k=self.locality, n_qubits=self.n_qubits)
        linear_combination = [sum(pauli_strings)]
        observables = linear_combination

        self.w = tf.Variable(
            initial_value=tf.constant([.5, .5]), dtype="float32",
            trainable=True, name="obs-weights")
        coeff_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0)
        self.coeff = tf.Variable(
            initial_value=coeff_init(shape=(1, len(self.omegas)*2+1), dtype="float32"),
            trainable=True, name="coefficients", constraint= lambda x: (x * len(pauli_strings)) / np.sqrt(x.dot(x))
        )

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

            avg_rewards = np.mean(episode_reward_history[-10:])

            if self.breakout:
                if avg_rewards >= 500.0:
                    break

        if self.savename != False:
            self.save_data(episode_reward_history)


def main():
    '''
    Initializes all the hyperparameters, creates the base and target network by calling upon dnn.py, and trains and saves the model by calling upon the QRL() class.
    '''
    env_name = "CartPole-v1"

    n_qubits = 4  # Dimension of the state vectors in CartPole
    n_actions = 2  # Number of actions in CartPole
    locality = 3 # the k-locality of the observables
    qubits = cirq.GridQubit.rect(1, n_qubits)

    n_episodes = 500
    learning_rates = [0.01, 0.01, 0.01, 0.01]
    gamma = 1
    beta = 1.0

    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    batch_size = 10

    breakout = False

    savename = 'test'

    start = time.time()

    qrl = QRL(savename=savename, locality=locality, n_qubits=n_qubits, n_actions=n_actions,
              env_name=env_name, n_episodes=n_episodes, batch_size=batch_size, learning_rates=learning_rates,
              gamma=gamma, state_bounds=state_bounds, breakout=breakout)

    qrl.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))

if __name__ == '__main__':
    main()