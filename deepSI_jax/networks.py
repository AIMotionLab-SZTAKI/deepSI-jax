import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray
import numpy as np
import flax.linen as nn


epsilon_z = 1e-4  # small constant to avoid division by zero
tol_z = 1e-6   # tolerance for which z and v are considered zero


def set_default_net_struct_if_necessary(struct_dict: dict[str, int|str|bool]):
    """Sets default values for the ANN structure, if those were not provided.

    Args:
        struct_dict (dict) : Dictionary containing the ANN structure description, e.g., number of hidden layers, etc.
    """
    if 'hidden_layers' not in struct_dict:
        struct_dict['hidden_layers'] = 2
    if 'nodes_per_layer' not in struct_dict:
        struct_dict['nodes_per_layer'] = 16
    if 'activation' not in struct_dict:
        struct_dict['activation'] = 'tanh'
    if 'feedthrough' not in struct_dict:
        struct_dict['feedthrough'] = True


def initialize_weights_and_biases(layer_units: list, input_features: int, key: PRNGKeyArray):
    """Initializes the parameters of an ANN with the same default method as pytorch.

    Args:
        layer_units (list) : Listing the number of nodes (neurons) in the ANN.
        input_features (int) : Dimension of the input vector.
        key (PRNGKeyArray) : Random key for initialization.

    Returns:
        weights_and_bias (list) : Listing of the initialized parameter values for the weights and biases of the ANN.
    """

    if not jax.config.jax_enable_x64:
        # Enable 64-bit computations
        jax.config.update("jax_enable_x64", True)

    weights_and_bias = []
    key_carry, key_w, key_b = jax.random.split(key, 3)

    for i, units in enumerate(layer_units):
        if i == 0:
            # first layer weight has dim (num_units, input shape)
            init_bnd = np.sqrt(1 / input_features)
            w = jax.random.uniform(key=key_w, shape=(units, input_features), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
            b = jax.random.uniform(key=key_b, minval=-init_bnd, maxval=init_bnd, shape=(units,), dtype=jnp.float64)
        else:
            # if not first layer
            key_carry, key_w, key_b = jax.random.split(key_carry, 3)
            init_bnd = np.sqrt(1 / layer_units[i-1])
            w = jax.random.uniform(key=key_w, shape=(units, layer_units[i-1]), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
            b = jax.random.uniform(key=key_b, minval=-init_bnd, maxval=init_bnd, shape=(units,), dtype=jnp.float64)
        # append weights
        weights_and_bias.append(w)
        weights_and_bias.append(b)
    # add residual component: weight has (output dim, input_dim) shape and no bias is applied
    init_bnd = np.sqrt(1 / input_features)
    w = jax.random.uniform(key=key_carry, shape=(layer_units[-1], input_features), minval=-init_bnd, maxval=init_bnd, dtype=jnp.float64)
    weights_and_bias.append(w)
    return weights_and_bias


def initialize_network(input_features: int, output_features: int, hidden_layers: int, nodes_per_layer: int, key: PRNGKeyArray):
    """Creates list containing the number of nodes per layer in the ANN, then calls ``initialize_weights_and_biases``
    to initialize the parameters of the network.

    Args:
        input_features (int) : Dimension of the input vector.
        output_features (int) : Dimension of the output vector.
        hidden_layers (int) : Number of hidden layers.
        nodes_per_layer (int) : Number of nodes per layer regarding the hidden layers..
        key (PRNGKeyArray) : random key for initialization.

    Returns:
        The initialized parameters of the ANN.
    """
    # list network architecture
    net_units = [nodes_per_layer]
    for i in range(hidden_layers-1):
        net_units.append(nodes_per_layer)
    net_units.append(output_features)
    parameters = initialize_weights_and_biases(layer_units=net_units, input_features=input_features, key=key)
    return parameters


relu = lambda x: jnp.maximum(jnp.zeros_like(a=x), x)


def activation(x, activation=None):
    """Dense ANN layer with optional activation function.

    Args:
        x (ndarray) : Layer input.
        activation (string, optional) : Activation function type. Possible values: 'relu' for rectified linear,
            'tanh' for hyperbolic tangent, 'sigmoid' for logistic, 'swish' or None for no activation (linear).
            (default: None)

    Returns:
        y (ndarray) : y = s(x), where is the selected activation function.
    """
    if activation == None:
        y =  x
    elif activation == 'relu':
        y = relu(x)
    elif activation == 'tanh':
        y = jnp.tanh(x)
    elif activation == 'sigmoid':
        y = nn.sigmoid(x)
    elif activation == 'swish':
        y = nn.swish(x)
    else:
        raise NotImplementedError('Further activation functions should be implemented by user!')
    return y


def generate_simple_res_net(idx_start: int, hidden_layers: int, act_fun: str):
    def net(net_in, params):
        W = params[idx_start]
        b = params[idx_start+1]
        y_next = activation(net_in @ W.T + b, act_fun)
        for i in range(hidden_layers - 1):
            W = params[idx_start + 2*i + 2]
            b = params[idx_start + 2*i + 3]
            y_next = activation(y_next @ W.T + b, act_fun)
        W = params[idx_start + 2*hidden_layers]
        b = params[idx_start + 2*hidden_layers + 1]
        W_res = params[idx_start + 2*hidden_layers + 2]
        # output with linear activation and residual component
        y_out = y_next @ W.T + b + net_in @ W_res.T
        return y_out
    return net


def gen_f_h_encoder_networks(nx: int, ny: int, nu: int, encoder_lag: int, f_args: dict[str, int|str],
                             h_args: dict[str, int|str|bool], encoder_args: dict[str, int|str], seed=0):
    """Creates and initializes the encoder network, state transition network, and the output network.

    Args:
        nx (int) : Dimension of the state vector.
        ny (int) : Dimension of the output vector.
        nu (int) : Dimension of the input vector.
        encoder_lag (int) : Number of past IO data points used for state reconstruction.
        f_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition function.
            Dictionary entries are 'hidden_layers' specifying the number of hidden layers, 'nodes_per_layer' specifying
            the applied number of nodes (neurons) per layer, and 'activation' for the activation function.
        h_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map function.
            Dictionary entries are the same as for ``f_args``, with the addition of 'feedthrough' (bool) which
            indicates weather the current output values depends on the current input value (True) or not (False).
        encoder_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the encoder.
            Dictionary entries are the same as for ``f_args``
        seed (int, optional) : Seed for initialization. (default: 0)

    Returns:
        f_net (function) : State transition ANN function.
        h_net (function) : Output map ANN function.
        encoder_net (function) : Encoder ANN function.
        params (list of ndararys) : List containing the combined initial values for all ANNs in the model.
    """

    f_ann = generate_simple_res_net(0, f_args['hidden_layers'], f_args['activation'])
    h_idx0 = 2 * f_args['hidden_layers'] + 3
    h_ann = generate_simple_res_net(h_idx0, h_args['hidden_layers'], h_args['activation'])
    enc_idx0 = h_idx0 + 2 * h_args['hidden_layers'] + 3
    enc_ann = generate_simple_res_net(enc_idx0, encoder_args['hidden_layers'], encoder_args['activation'])

    @jax.jit
    def f_net(x, u, params):
        # f : (nx, nu) --> nx
        net_in = jnp.hstack((x, u))
        return f_ann(net_in, params)

    @jax.jit
    def h_net(x, u, params):
        # f : (nx, nu) --> ny OR nx --> ny
        if h_args['feedthrough']:
            net_in = jnp.hstack((x, u))
        else:
            net_in = x
        return h_ann(net_in, params)

    @jax.jit
    def encoder_net(yu_hist, params):
        # e : (n*ny+n*nu) --> nx
        return enc_ann(yu_hist, params)

    key = jax.random.key(seed)
    key_f, key_h, key_enc = jax.random.split(key, 3)

    f_input_dim = nx + nu
    f_init_params = initialize_network(input_features=f_input_dim, output_features=nx, hidden_layers=f_args['hidden_layers'],
                                       nodes_per_layer=f_args['nodes_per_layer'], key=key_f)
    if h_args['feedthrough']:
        h_input_dim = nx + nu
    else:
        h_input_dim = nx
    h_init_params = initialize_network(input_features=h_input_dim, output_features=ny, hidden_layers=h_args['hidden_layers'],
                                       nodes_per_layer=h_args['nodes_per_layer'], key=key_h)
    encoder_init_params = initialize_network(input_features=encoder_lag*(ny+nu), output_features=nx, hidden_layers=encoder_args['hidden_layers'],
                                             nodes_per_layer=encoder_args['nodes_per_layer'], key=key_enc)

    params = f_init_params
    params.extend(h_init_params)
    params.extend(encoder_init_params)

    return f_net, h_net, encoder_net, params


def gen_f_h_networks(nx: int, ny: int, nu: int, f_args: dict[str, int|str], h_args: dict[str, int|str|bool], seed=0):
    """Creates and initializes the encoder network, state transition network, and the output network.

    Args:
        nx (int) : Dimension of the state vector.
        ny (int) : Dimension of the output vector.
        nu (int) : Dimension of the input vector.
        f_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition function.
            Dictionary entries are 'hidden_layers' specifying the number of hidden layers, 'nodes_per_layer' specifying
            the applied number of nodes (neurons) per layer, and 'activation' for the activation function.
        h_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map function.
            Dictionary entries are the same as for ``f_args``, with the addition of 'feedthrough' (bool) which
            indicates weather the current output values depends on the current input value (True) or not (False).
        seed (int, optional) : Seed for initialization. (default: 0)

    Returns:
        f_net (function) : State transition ANN function.
        h_net (function) : Output map ANN function.
        params (list of ndararys) : List containing the combined initial values for all ANNs in the model.
    """

    f_ann = generate_simple_res_net(0, f_args['hidden_layers'], f_args['activation'])
    h_idx0 = 2 * f_args['hidden_layers'] + 3
    h_ann = generate_simple_res_net(h_idx0, h_args['hidden_layers'], h_args['activation'])

    @jax.jit
    def f_net(x, u, params):
        # f : (nx, nu) --> nx
        net_in = jnp.hstack((x, u))
        return f_ann(net_in, params)

    @jax.jit
    def h_net(x, u, params):
        # f : (nx, nu) --> ny OR nx --> ny
        if h_args['feedthrough']:
            net_in = jnp.hstack((x, u))
        else:
            net_in = x
        return h_ann(net_in, params)

    key = jax.random.key(seed)
    key_f, key_h = jax.random.split(key, 2)

    f_input_dim = nx + nu
    f_init_params = initialize_network(input_features=f_input_dim, output_features=nx, hidden_layers=f_args['hidden_layers'],
                                       nodes_per_layer=f_args['nodes_per_layer'], key=key_f)
    if h_args['feedthrough']:
        h_input_dim = nx + nu
    else:
        h_input_dim = nx
    h_init_params = initialize_network(input_features=h_input_dim, output_features=ny, hidden_layers=h_args['hidden_layers'],
                                       nodes_per_layer=h_args['nodes_per_layer'], key=key_h)

    params = f_init_params
    params.extend(h_init_params)

    return f_net, h_net, params
