from quantum_model import *
from argparse import ArgumentParser

env_name = "CartPole-v1"
flipped_model = True # whether to use the flipped model or the non-flipped model
# amount of repetitions that will be averaged over for the experiment
repetitions = 20
# amount of episodes that will run
n_episodes = 2000
n_qubits = 4
n_actions = 2
locality = 3 # the k-locality of the observables

qubits = cirq.GridQubit.rect(1, n_qubits)

if flipped_model:
    n_layers = 1  # Number of variational layers in the PQC. Note that currently it only applies a Y and Z rotation
    #ops = [cirq.I(q) for q in qubits]
    #ops[0], ops[-1] = cirq.Z(qubits[0]), cirq.Z(qubits[-1]) # Z_0*I*I*Z_3
    #observables = [reduce((lambda x, y: x * y), ops)]
    pauli_strings = get_k_local(k=locality, n_qubits=n_qubits)
    linear_combination = [sum(pauli_strings)]
    observables = linear_combination
else:
    n_layers = 5  # Number of layers in the PQC
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3

# Hyperparameters of the algorithm and other parameters of the program
learning_rate_in = [0.5, 0.1, 0.01]
learning_rate_var = [0.1, 0.01, 0.001]
learning_rate_out = [0.5, 0.1, 0.01]
gamma = 1  # discount factor
batch_size = 10
beta = 1.0
state_bounds = np.array([2.4, 2.5, 0.21, 2.5])

breakout = False

data_names = []

start = time.time()

argparser = ArgumentParser()
argparser.add_argument("savename", default="lr_in_", nargs="?")
args = argparser.parse_args()
savename = args.savename

for lr_in in learning_rate_in:
    for lr_var in learning_rate_var:
        for lr_out in learning_rate_out:
            for rep in range(repetitions):
                file_name = savename + str(lr_in) + '-lr_var_' + str(lr_var) + '-lr_out_' + str(lr_out) + '-repetition_' + str(rep + 1)

                quantum_model = QuantumModel(qubits=qubits, n_layers=n_layers, observables=observables)

                model = quantum_model.generate_model_policy(n_actions=n_actions, beta=beta, flipped_model=flipped_model)

                qrl = QRL(savename=file_name, model=model, learning_rates=[lr_in, lr_var, lr_out], gamma=gamma, n_episodes=n_episodes,
                          batch_size=batch_size, state_bounds=state_bounds,
                          n_qubits=n_qubits, n_layers=n_layers, n_actions=n_actions, env_name=env_name, breakout=breakout)

                qrl.main()

                data_names.append(file_name)

                print('Finished repetition '+str(rep+1)+'/'+str(repetitions))

            plot_averaged(data_names=data_names, show=False, savename=savename+str(lr_in)+'-lr_var_'+str(lr_var)+'-lr_out_'+str(lr_out), smooth=False)
            plot_averaged(data_names=data_names, show=False, savename=savename+str(lr_in)+'-lr_var_'+str(lr_var)+'-lr_out_'+str(lr_out)+'-smooth', smooth=True)

data_names = []

end = time.time()

print('Total time: {} seconds'.format(round(end - start, 1)))