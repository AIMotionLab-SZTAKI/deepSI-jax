import numpy as np
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed, cpu_count
from deepSI_jax.data_prep import create_ndarray_from_list
from jax_sysid.utils import vec_reshape


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
        Y = create_ndarray_from_list(Y.copy())
        Yhat = create_ndarray_from_list(Yhat.copy())
    Y = vec_reshape(Y.copy())
    Yhat = vec_reshape(Yhat.copy())
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
    rmse = RMS_error(Y.copy(), Yhat, per_channel=True)
    if isinstance(Y, list):
        y_std = np.std(create_ndarray_from_list(Y.copy()), axis=0)
    else:
        y_std = np.std(vec_reshape(Y.copy()), axis=0)
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


def find_best_model(models, Y, U, use_encoder=False, X0=None, n_jobs=None, verbose=True, fit="RMSE", seeds=None, use_training_x0=False):
    if not isinstance(models, list):
        raise Exception(
            "\033[1mPlease provide a list of models to compare.\033[0m")

    if len(models) == 1:
        return models[0]

    if fit.lower() == 'rmse':
        error_fun = RMS_error
    elif fit.lower() == 'nrmse':
        error_fun = NRMS_error
    elif fit.lower() == 'mse':
        error_fun = mean_squared_error

    if isinstance(Y, list):
        N_meas = len(Y)
    else:
        N_meas = 1

    if use_training_x0:
        X0 = []
        for i in range(len(models)):
            X0.append(models[i].x0)

    def get_error_no_encoder(k):
        if X0 is None and N_meas == 1:
            x0 = models[k].learn_x0(U, Y)
        elif X0 is None and N_meas > 1:
            x0 = []
            for i in range(N_meas):
                x0i = models[k].learn_x0(U[i], Y[i])
                x0.append(x0i)
        else:
            x0 = X0.copy()
        Yhat, _ = models[k].simulate(x0, U)
        return error_fun(Y, Yhat)

    def get_error_encoder(k):
        if N_meas == 1:
            Y_lag = vec_reshape(Y)[:models[k].encoder_lag, :]
            Y_true = vec_reshape(Y)[models[k].encoder_lag:, :]
            U_lag = vec_reshape(U)[:models[k].encoder_lag, :]
            x0 = models[k].encoder_estim_x0(Y_lag, U_lag)
            Yhat, _ = models[k].simulate(x0, vec_reshape(U)[models[k].encoder_lag:, :])
        else:  # N_meas > 1
            Y_lag = [vec_reshape(y)[:models[k].encoder_lag, :]for y in Y]
            Y_true = [vec_reshape(y)[models[k].encoder_lag:, :] for y in Y]
            U_lag = [vec_reshape(u)[:models[k].encoder_lag, :]for u in U]
            U_true = [vec_reshape(u)[models[k].encoder_lag:, :] for u in U]
            x0 = models[k].encoder_estim_x0(Y_lag, U_lag)
            Yhat, _ = models[k].simulate(x0, U_true)
        return error_fun(Y_true, Yhat)

    if n_jobs is None:
        n_jobs = cpu_count()  # Use all available cores by default

    if verbose:
        print("Evaluating models...\n")

    if use_encoder:
        errors = Parallel(n_jobs=n_jobs)(delayed(get_error_encoder)(k) for k in range(len(models)))
    else:
        errors = Parallel(n_jobs=n_jobs)(delayed(get_error_no_encoder)(k) for k in range(len(models)))

    # get the lowest error (lowest average error in case of multiple outputs)
    best_id = np.nanargmin(np.sum(np.array(errors).reshape(len(models), -1), axis=1))

    if verbose:
        print("Errors:")
        for k in range(len(models)):
            print(f"Model {k}: {fit} = {errors[k]}")
        if seeds is None:
            print(f"Best model: {best_id}, error: {errors[best_id]}")
        else:
            print(f"Best model: {best_id}, score: {errors[best_id]} at seed {seeds[best_id]}")

    return models[best_id]
