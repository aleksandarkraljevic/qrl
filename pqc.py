import tensorflow as tf
import tensorflow_quantum as tfq
import cirq, sympy
import numpy as np

class QuantumModel():
    def __init__(self, qubits, n_layers, observables):
        '''
        Initializes the parameters for the PQC.

        Parameters
        ----------
        qubits (int):
            The number of qubits that the PQC will use.
        n_layers (int):
            The number of layers that the PQC will contain.
        observables (list):
            The list of the observables that will be measured.
        '''
        self.qubits = qubits
        self.n_layers = n_layers
        self.observables = observables

    def one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in `symbols`.
        """
        return [cirq.rx(symbols[0])(qubit),
                cirq.ry(symbols[1])(qubit),
                cirq.rz(symbols[2])(qubit)]

    def reduced_one_qubit_rotation(self, qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the
        Y and Z axis, specified by the values in `symbols`.
        """
        return [cirq.ry(symbols[0])(qubit),
                cirq.rz(symbols[1])(qubit)]

    def entangling_layer(self):
        """
        Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
        """
        cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(self.qubits, self.qubits[1:])]
        cz_ops += ([cirq.CZ(self.qubits[0], self.qubits[-1])] if len(self.qubits) != 2 else [])
        return cz_ops

    def generate_circuit(self):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
        # Number of qubits
        n_qubits = len(self.qubits)

        # Sympy symbols for variational angles
        params = sympy.symbols(f'theta(0:{3*(self.n_layers+1)*n_qubits})')
        params = np.asarray(params).reshape((self.n_layers + 1, n_qubits, 3))

        # Sympy symbols for encoding angles
        inputs = sympy.symbols(f'x(0:{self.n_layers})'+f'_(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((self.n_layers, n_qubits))

        # Define circuit
        circuit = cirq.Circuit()
        for l in range(self.n_layers):
            # Variational layer
            circuit += cirq.Circuit(self.one_qubit_rotation(q, params[l, i]) for i, q in enumerate(self.qubits))
            circuit += self.entangling_layer()
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(self.qubits))

        # Last varitional layer
        circuit += cirq.Circuit(self.one_qubit_rotation(q, params[self.n_layers, i]) for i,q in enumerate(self.qubits))

        return circuit, list(params.flat), list(inputs.flat)

    def generate_flipped_circuit(self):
        """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
        # Number of qubits
        n_qubits = len(self.qubits)

        # Sympy symbols for variational angles of the first layer
        params_layer_one = sympy.symbols(f'theta(0:{2*n_qubits})')
        params_layer_one = np.asarray(params_layer_one).reshape((1, n_qubits, 2))

        # Sympy symbols for variational angles
        params = sympy.symbols(f'theta({2*n_qubits}:{2*n_qubits+3*(self.n_layers-1)*n_qubits})')
        params = np.asarray(params).reshape((self.n_layers-1, n_qubits, 3))

        # Sympy symbols for encoding angles
        inputs = sympy.symbols(f'x(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((1, n_qubits))

        # Define circuit
        circuit = cirq.Circuit()
        # First variational layer (YZ)
        circuit += cirq.Circuit(self.reduced_one_qubit_rotation(q, params_layer_one[0, i]) for i, q in enumerate(self.qubits))
        circuit += self.entangling_layer()
        for l in range(self.n_layers-1): # A layer contains only variational parts
            # Variational layer (XYZ)
            circuit += cirq.Circuit(self.one_qubit_rotation(q, params[l, i]) for i, q in enumerate(self.qubits))
            circuit += self.entangling_layer()
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[0, i])(q) for i, q in enumerate(self.qubits))

        params_layer_one_list = list(params_layer_one.flat)
        params_list = list(params.flat)
        params_list = params_layer_one_list + params_list

        return circuit, params_list, list(inputs.flat)

    def generate_model_policy(self, n_actions, beta):
        """Generates a Keras model for a data re-uploading PQC policy."""

        input_tensor = tf.keras.Input(shape=(len(self.qubits),), dtype=tf.dtypes.float32, name='input')
        re_uploading_pqc = ReUploadingPQC(self.qubits, self.n_layers, self.observables)([input_tensor])
        process = tf.keras.Sequential([
            Alternating(n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(re_uploading_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

        return model

    def generate_flipped_model_policy(self, n_actions, beta):
        """Generates a Keras model for a data flipped PQC policy."""

        input_tensor = tf.keras.Input(shape=(len(self.qubits),), dtype=tf.dtypes.float32, name='input')
        flipped_pqc = FlippedPQC(self.qubits, self.n_layers, self.observables)([input_tensor])
        process = tf.keras.Sequential([
            Alternating(n_actions),
            tf.keras.layers.Lambda(lambda x: x * beta),
            tf.keras.layers.Softmax()
        ], name="observables-policy")
        policy = process(flipped_pqc)
        model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

        return model

class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        quantum_model = QuantumModel(qubits, n_layers, observables)
        circuit, theta_symbols, input_symbols = quantum_model.generate_circuit()

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

class FlippedPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1,
        ......., lmbd[d][1]s_d) for d=input_dim and N=theta_dim.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="flipped_PQC"):
        super(FlippedPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        quantum_model = QuantumModel(qubits, n_layers, observables)
        circuit, theta_symbols, input_symbols = quantum_model.generate_flipped_circuit()

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])
