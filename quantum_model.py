import importlib, pkg_resources
importlib.reload(pkg_resources)

from helper import *

import gym, cirq, sympy
import time
import numpy as np
from collections import deque, defaultdict
from functools import reduce
tf.get_logger().setLevel('ERROR')

class QRL():
    def __init__(self, savename, model, learning_rates, gamma, n_episodes, batch_size, state_bounds, n_qubits, n_layers, n_actions, env_name):
        '''
        Initializes the QRL parameters.

        Parameters
        ----------
        savename (str):
            The name with which the file model and data files will be saved.
        model (tensorflow keras model):
            The base network.
        model_target (tensorflow keras model):
            The target network.
        n_layers (int):
            The number of layers that the PQC will contain.
        n_holes (int):
            Number of outputs of the DNN. / Number of holes in the environment.
        qubits (int):
            The number of qubits that the PQC will use.
        memory_size (int):
            The amount of guesses that the agent is allowed to look back. / The state-space size.
        learning_rates (list):
            A list of three learning rates that the optimizers within the PQC use in order to update the encoding, variational, and rescaling weights.
        gamma (float):
            The discount factor that is used in the Q-learning algorithm.
        n_episodes (int):
            The amount of total episodes that the model will train for.
        steps_per_train (int):
            The amount of time steps that pass in between each training step.
        soft_weight_update (boolean):
            Whether the target network will be updated via soft-updating. If False, then it will be hard-updating.
        steps_per_target_update (int):
            Per how many training steps the target network will update, if it is hard-updating.
        tau (float):
            The fraction with which the base network copies over to the target network after each training step.
        epsilon_start (float):
            The starting value of epsilon in the case that annealing epsilon-greedy is used.
        epsilon_min (float):
            The lowest value of epsilon in the case that annealing epsilon-greedy is used.
        decay_epsilon (float):
            How fast epsilon decays in the case that annealing epsilon-greedy is used.
        temperature (float):
            The strength with which exploration finds place in the case that the Boltzmann policy is used.
        batch_size (int):
            The amount of samples that each training batch consists of.
        min_size_buffer (int):
            The minimum size that the experience replay buffer needs to be before training may start.
        max_size_buffer (int):
            The maximum size that the experience replay buffer is allowed to be. If this limit is reached then the oldest samples start being replaced with the newest samples.
        exploration_strategy (str):
            What exploration strategy should be followed during the training of the model. Either "egreedy" or "boltzmann".
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
        # Indexes of the weights for each of the parts of the circuit
        self.w_in, self.w_var, self.w_out = 1, 0, 2

    def gather_episodes(self):
        """Interact with environment in batched fashion."""

        trajectories = [defaultdict(list) for _ in range(self.n_episodes)]
        envs = [gym.make(self.env_name) for _ in range(self.n_episodes)]

        done = [False for _ in range(self.n_episodes)]
        states = [e.reset() for e in envs]

        while not all(done):
            unfinished_ids = [i for i in range(self.n_episodes) if not done[i]]
            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)

            # Compute policy for all unfinished envs in parallel
            states = tf.convert_to_tensor(normalized_states)
            action_probs = self.model([states])

            # Store action and transition all environments to the next state
            states = [None for i in range(self.n_episodes)]
            for i, policy in zip(unfinished_ids, action_probs.numpy()):
                action = np.random.choice(self.n_actions, p=policy)
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

    @tf.function
    def reinforce_update(self, states, actions, returns):
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

    def main(self):
        # Start training the agent
        episode_reward_history = []
        for batch in range(self.n_episodes // self.batch_size):
            # Gather episodes
            episodes = self.gather_episodes()

            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds, self.gamma) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            # Update model parameters.
            self.reinforce_update(states, id_action_pairs, returns, self.model)

            # Store collected rewards
            for ep_rwds in rewards:
                episode_reward_history.append(np.sum(ep_rwds))

            avg_rewards = np.mean(episode_reward_history[-10:])


def main():
    '''
    Initializes all the hyperparameters, creates the base and target network by calling upon dnn.py, and trains and saves the model by calling upon the QRL() class.
    '''
    env_name = "CartPole-v1"

    n_qubits = 4  # Dimension of the state vectors in CartPole
    n_layers = 5  # Number of layers in the PQC
    n_actions = 2  # Number of actions in CartPole

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3

    n_episodes = 5000
    learning_rates = [0.1, 0.01, 0.1]
    gamma = 1
    beta = 1.0

    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    batch_size = 64

    savename = 'test'

    start = time.time()

    quantum_model = QuantumModel(qubits, n_layers, observables)

    model = quantum_model.generate_model_policy(n_actions=n_actions, beta=beta)

    qrl = QRL(savename=savename, model=model, learning_rates=learning_rates, gamma=gamma, n_episodes=n_episodes, batch_size=batch_size, state_bounds=state_bounds,
              n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions, env_name=env_name)

    qrl.main()

    end = time.time()

    print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))

if __name__ == '__main__':
    main()