import numpy as np
import jax
import jax.numpy as jnp
from jax_sysid.utils import vec_reshape


def mean_squared_error(Y, Yhat, per_channel=False):
    Y = vec_reshape(Y)
    Yhat = vec_reshape(Yhat)
    error = Y - Yhat  # (N, ny)
    mse = np.mean(error**2, axis=0)
    if not per_channel:
        mse = np.mean(mse)
    return mse

def RMS_error(Y, Yhat, per_channel=False):
    return mean_squared_error(Y, Yhat, per_channel=per_channel)**0.5

def NRMS_error(Y, Yhat, per_channel=False):
    rmse = RMS_error(Y, Yhat, per_channel=True)
    y_std = np.std(vec_reshape(Y), axis=0)
    nrmse = rmse / y_std
    if not per_channel:
        nrmse = np.mean(nrmse)
    return nrmse


@jax.jit
def xsat(x, sat):
    """
    Apply saturation to the state value.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The state value.
    sat : float
        The saturation limit.

    Returns
    -------
    jax.numpy.ndarray
        The saturated value of x.
    """
    if sat is None:
        return x
    else:
        return jnp.minimum(jnp.maximum(x, -sat), sat)  # hard saturation
