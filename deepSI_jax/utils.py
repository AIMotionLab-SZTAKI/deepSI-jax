import numpy as np
import jax
import jax.numpy as jnp
from jax_sysid.utils import vec_reshape


def create_numpy_array_from_multiple_experiments(var_list: list) -> np.ndarray:
    """Creates a numpy array from multiple experiments.

    Args:
        var_list (list) : List of Ni-by-n numpy arrays, where Ni is the length of the i-th experiment.

    Returns:
        var_array (ndarray) : A combined numpy array with all the experiments with shape N-by-n, where N = sum Ni.
    """
    var_array = np.array([])
    for i in range(len(var_list)):
        var_i = vec_reshape(var_list[i])
        if i == 0:
            var_array = var_i
        else:
            var_array = np.vstack((var_array, var_i))
    return var_array


def mean_squared_error(Y: np.ndarray|list, Yhat: np.ndarray|list, per_channel=False):
    """Computes the mean-squared error (MSE) between the predicted (simulated) and true output.

    Args:
        Y (ndarray or list of ndarrays) : True measured output with shape N-by-ny or a list of Ni-by-ny arrays.
        Yhat (ndarray or list of ndarrays) : Predicted (simulated) output values with shape N-by-ny or a list of Ni-by-ny arrays.
        per_channel (bool, optional) : If True, returns the mean squared error per channel. Defaults to False.

    Returns:
        mse (float or ndarray) : The mean-squared error between Yhat and Y.
    """
    if isinstance(Y, list):
        Y = create_numpy_array_from_multiple_experiments(Y)
        Yhat = create_numpy_array_from_multiple_experiments(Yhat)
    Y = vec_reshape(Y)
    Yhat = vec_reshape(Yhat)
    error = Y - Yhat  # (N, ny)
    mse = np.mean(error**2, axis=0)
    if not per_channel:
        mse = np.mean(mse)
    return mse


def RMS_error(Y: np.ndarray|list, Yhat: np.ndarray|list, per_channel=False):
    """Computes the root-mean-squared error (RMSE) between the predicted (simulated) and true output.

    Args:
        Y (ndarray or list of ndarrays) : True measured output with shape N-by-ny or a list of Ni-by-ny arrays.
        Yhat (ndarray or list of ndarrays) : Predicted (simulated) output values with shape N-by-ny or a list of Ni-by-ny arrays.
        per_channel (bool, optional) : If True, returns the mean squared error per channel. Defaults to False.

    Returns:
        The root-mean-squared error between Yhat and Y.
    """
    return mean_squared_error(Y, Yhat, per_channel=per_channel)**0.5


def NRMS_error(Y: np.ndarray|list, Yhat: np.ndarray|list, per_channel=False):
    """Computes the normalized root-mean-squared error (NRMSE) between the predicted (simulated) and true output.
    The root-mean-squared error is normalized with the standard deviation of the output signal Y.

    Args:
        Y (ndarray or list of ndarrays) : True measured output with shape N-by-ny or a list of Ni-by-ny arrays.
        Yhat (ndarray or list of ndarrays) : Predicted (simulated) output values with shape N-by-ny or a list of Ni-by-ny arrays.
        per_channel (bool, optional) : If True, returns the mean squared error per channel. Defaults to False.

    Returns:
        nrmse (float or ndarray) : The normalized root-mean-squared error between Yhat and Y.
    """
    rmse = RMS_error(Y, Yhat, per_channel=True)
    if isinstance(Y, list):
        y_std = np.std(create_numpy_array_from_multiple_experiments(Y), axis=0)
    else:
        y_std = np.std(vec_reshape(Y), axis=0)
    nrmse = rmse / y_std
    if not per_channel:
        nrmse = np.mean(nrmse)
    return nrmse


@jax.jit
def xsat(x: jnp.ndarray, sat: float|None):
    """Apply saturation for the state values in simulation (training) only if necessary.

    Args:
        x (jnp.ndarray): state vector.
        sat (float or None): If None, no saturation is applied, otherwise the state value is saturated at sat value.

    Return:
        Saturated state value.
    """
    if sat is None:
        return x
    else:
        return jnp.minimum(jnp.maximum(x, -sat), sat)  # hard saturation
