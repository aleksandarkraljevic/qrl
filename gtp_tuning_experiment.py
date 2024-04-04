from gtp import *
from argparse import ArgumentParser

EXPERIMENTS = [
    #("lr_coeff, lr_out"),
    (0.1, 0.1),
    (0.1, 0.01),
    (0.1, 0.001),
    (0.1, 0.0001),
    (0.01, 0.1),
    (0.01, 0.01),
    (0.01, 0.001),
    (0.01, 0.0001),
    (0.001, 0.1),
    (0.001, 0.01),
    (0.001, 0.001),
    (0.001, 0.0001),
    (0.0001, 0.1),
    (0.0001, 0.01),
    (0.0001, 0.001),
    (0.0001, 0.0001),
]


argparser = ArgumentParser()
#argparser.add_argument("savename", default="lr_in_", nargs="?")
argparser.add_argument("--batch_n", type=int)
args = argparser.parse_args()
#savename = args.savename
savename = 'lr_coeff_'
experiment = EXPERIMENTS[args.batch_n]

env_name = "CartPole-v1"
# amount of repetitions that will be averaged over for the experiment
repetitions = 20
# amount of episodes that will run
n_episodes = 2000
n_qubits = 4
n_actions = 2
locality = 3 # the k-locality of the observables

pauli_strings = get_k_local(k=locality, n_qubits=n_qubits)
linear_combination = [sum(pauli_strings)]
observables = linear_combination

# Hyperparameters of the algorithm and other parameters of the program
learning_rate_coeff = experiment[0]
learning_rate_out = experiment[1]
gamma = 1  # discount factor
batch_size = 10
beta = 1.0
state_bounds = np.array([2.4, 2.5, 0.21, 2.5])

breakout = False

data_names = []

start = time.time()

for rep in range(repetitions):
    parameter_savename = str(learning_rate_coeff) + '-lr_out_' + str(learning_rate_out)
    file_name = savename + parameter_savename + '-repetition_' + str(rep + 1)

    qrl = GTP_QRL(savename=file_name, locality=locality, n_qubits=n_qubits, n_actions=n_actions,
                  env_name=env_name, n_episodes=n_episodes, batch_size=batch_size, learning_rates=[learning_rate_coeff, learning_rate_out],
                  gamma=gamma, beta=beta, state_bounds=state_bounds, breakout=breakout)

    qrl.main()

    data_names.append(file_name)

    print('Finished repetition '+str(rep+1)+'/'+str(repetitions))

plot_averaged(data_names=data_names, show=False, savename=savename+parameter_savename, smooth=False)
plot_averaged(data_names=data_names, show=False, savename=savename+parameter_savename+'-smooth', smooth=True)

data_names = []

end = time.time()

print('Total time: {} seconds'.format(round(end - start, 1)))