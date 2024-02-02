from quantum_model import *

env_name = "CartPole-v1"
# amount of repetitions that will be averaged over for the experiment
repetitions = 2
# amount of episodes that will run
n_episodes = 500
n_qubits = 4
n_layers = 5  # Number of layers in the PQC
n_actions = 2

qubits = cirq.GridQubit.rect(1, n_qubits)
ops = [cirq.Z(q) for q in qubits]
observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3

# Hyperparameters of the algorithm and other parameters of the program
learning_rate_in = 0.1
learning_rate_var = 0.01
learning_rate_out = 0.1
gamma = 1  # discount factor
batch_size = 10
beta = 1.0
state_bounds = np.array([2.4, 2.5, 0.21, 2.5])

breakout = False

data_names = []

start = time.time()

savename = 'test_experiment'
for rep in range(repetitions):
    file_name = savename + '-repetition_' + str(rep + 1)

    quantum_model = QuantumModel(qubits, n_layers, observables)

    model = quantum_model.generate_model_policy(n_actions=n_actions, beta=beta)

    qrl = QRL(savename=file_name, model=model, learning_rates=[learning_rate_in, learning_rate_var, learning_rate_out], gamma=gamma, n_episodes=n_episodes,
              batch_size=batch_size, state_bounds=state_bounds,
              n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions, env_name=env_name, breakout=breakout)

    qrl.main()

    data_names.append(file_name)

    print('Finished repetition '+str(rep+1)+'/'+str(repetitions))

plot_averaged(data_names=data_names, show=False, savename=savename, smooth=False)
plot_averaged(data_names=data_names, show=False, savename=savename+'-smooth', smooth=True)

data_names = []

end = time.time()

print('Total time: {} seconds (number of episodes: {})'.format(round(end - start, 1), n_episodes))