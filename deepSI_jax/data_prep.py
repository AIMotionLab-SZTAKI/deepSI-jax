import numpy as np
from jax_sysid.utils import vec_reshape


def create_multi_shooting_data_single(Y: np.ndarray, U: np.ndarray, T: int, overlap: int):
    """Divides the training data into multiple truncated sections (without any overlap).

    If N (number of data points) is not dividable by T, the last T data points are also added to the list,
    with a possible overlap.

    Args:
        Y (ndarray) : N-by-ny array containing the output values of the training data.
        U (ndarray) : N-by-nu array containing the input values of the training data.
        T (int) : Truncation length (length of the sections).
        overlap (int) : Overlap between subsections of the training data.

    Returns:
        Yms (list of ndarrays) : A list containing the output values for the truncated sections with sizes T-by-ny.
        Ums (list of ndarrays) : A list containing the input values for the truncated sections with sizes T-by-nu.
    """
    Yms = []
    Ums = []

    Y = vec_reshape(Y)
    U = vec_reshape(U)

    N = Y.shape[0]
    num_sections = int(N / (T - overlap))
    idx_start = np.arange(num_sections) * (T - overlap)
    if idx_start[-1] > N - T:
        # if the last section would stick out of the data set --> adjust it accordingly
        idx_start[-1] = N - T

    for idx in idx_start:
        Yms.append(Y[idx:idx+T, :])
        Ums.append(U[idx:idx+T, :])

    return Yms, Ums


def create_multi_shooting_data(Y: np.ndarray | list, U:np.ndarray | list, T: int, overlap=0):
    """Divides the training data into multiple truncated sections (without any overlap).

    If only a single measurement record is provided, simply calls ``create_multi_shooting_data_single``.

    If N (number of data points) is not dividable by T, the last T data points are also added to the list,
    with a possible overlap.

    Args:
        Y (ndarray or list of ndarrays) : N-by-ny array containing the output values of the training data or a list of
            N-by-ny arrays.
        U (ndarray or list of ndarrays) : N-by-nu array containing the input values of the training data or a list of
            N-by-nu arrays.
        T (int) : Truncation length (length of the sections).
        overlap (int, optional) : Overlap between subsections of the training data. (default: 0)

    Returns:
        Yms (list of ndarrays) : A list containing the output values for the truncated sections with sizes T-by-ny.
        Ums (list of ndarrays) : A list containing the input values for the truncated sections with sizes T-by-nu.
    """
    if isinstance(Y, list):
        Nexp = len(Y)
        Yms = []
        Ums = []
        for i in range(Nexp):
            Ymsi, Umsi = create_multi_shooting_data_single(Y[i], U[i], T, overlap)
            Yms.append(Ymsi)
            Ums.append(Umsi)
    else:
        Yms, Ums = create_multi_shooting_data_single(Y, U, T, overlap)
    return Yms, Ums


def create_multi_shooting_data_with_hist_single(Y: np.ndarray, U: np.ndarray, T: int, n: int, overlap: int):
    """Divides the training data into multiple truncated sections (without any overlap) and creates an IO
    history sequence for state reconstruction with length n.

    If N (number of data points) is not dividable by T, the last T data points are also added to the list,
    with a possible overlap.

    Args:
        Y (ndarray) : N-by-ny array containing the output values of the training data.
        U (ndarray) : N-by-nu array containing the input values of the training data.
        T (int) : Truncation length (length of the sections).
        n (int) : Number of data points for state reconstruction (encoder lag).
        overlap (int) : Overlap between subsections of the training data.

    Returns:
        Yms (list of ndarrays) : A list containing the output values for the truncated sections with sizes T-by-ny.
        Ums (list of ndarrays) : A list containing the input values for the truncated sections with sizes T-by-nu.
        Yhist (list of ndarrays): A list containing the output histories for state reconstruction with sizes n-by-ny.
        Uhist (list of ndarrays): A list containing the input histories for state reconstruction with sizes n-by-nu.
    """
    Yms = []
    Yhist = []
    Uhist = []
    Ums = []

    Y = vec_reshape(Y)
    U = vec_reshape(U)

    N = Y.shape[0]
    num_sections = int((N - n) / (T - overlap))
    idx_start = np.arange(num_sections) * (T - overlap)
    if idx_start[-1] > N - n - T:
        # if the last section would stick out of the data set --> adjust it accordingly
        idx_start[-1] = N - n - T

    for idx in idx_start:
        Yhist.append(Y[idx:idx+n, :])
        Uhist.append(U[idx:idx+n, :])
        Yms.append(Y[idx+n:idx+n+T, :])
        Ums.append(U[idx+n:idx+n+T, :])

    return Yms, Ums, Yhist, Uhist


def create_multi_shooting_data_with_hist(Y: np.ndarray | list, U: np.ndarray | list, T: int, n: int, overlap=0):
    """Divides the training data into multiple truncated sections (without any overlap) and creates an IO
    history sequence for state reconstruction with length n.

    If only a single measurement record is provided, simply calls ``create_multi_shooting_data_with_hist_single``.

    If N (number of data points) is not dividable by T, the last T data points are also added to the list,
    with a possible overlap.

    Args:
        Y (ndarray or list of ndarrays) : N-by-ny array containing the output values of the training data or a list of
            N-by-ny arrays.
        U (ndarray or list of ndarrays) : N-by-nu array containing the input values of the training data or a list of
            N-by-nu arrays.
        T (int) : Truncation length (length of the sections).
        n (int) : Number of data points for state reconstruction (encoder lag).
        overlap (int, optional) : Overlap between subsections of the training data. (default: 0)

    Returns:
        Yms (list of ndarrays) : A list containing the output values for the truncated sections with sizes T-by-ny.
        Ums (list of ndarrays) : A list containing the input values for the truncated sections with sizes T-by-nu.
        Yhist (list of ndarrays): A list containing the output histories for state reconstruction with sizes n-by-ny.
        Uhist (list of ndarrays): A list containing the input histories for state reconstruction with sizes n-by-nu.
    """

    if overlap < 0 or overlap > T:
        raise ValueError('Overlap must be between 0 and T')

    if isinstance(Y, list):
        Nexp = len(Y)
        Yms = []
        Ums = []
        Yhist = []
        Uhist = []
        for i in range(Nexp):
            Ymsi, Umsi, Yhisti, Uhisti = create_multi_shooting_data_with_hist_single(Y[i], U[i], T, n, overlap)
            Yms.extend(Ymsi)
            Ums.extend(Umsi)
            Yhist.extend(Yhisti)
            Uhist.extend(Uhisti)
    else:
        Yms, Ums, Yhist, Uhist = create_multi_shooting_data_with_hist_single(Y, U, T, n, overlap)
    YU_hist = prepare_hist_data(Yhist, Uhist)
    return Yms, Ums, YU_hist


def prepare_hist_data(Yhist: list, Uhist: list):
    """Concatenates the input and output histories of the training data considering all truncated subsections for
    state reconstruction.

    Args:
        Yhist (list of ndarrays) : List of containing n-by-ny arrays of measured outputs.
        Uhist (list of ndarrays) : List of containing n-by-nu arrays of the inputs.

    Returns:
        YU_hist (ndarray) : Numpy array of the concatenated IO state reconstruction histories with size
            N-by-(n*ny+n*nu), where N is the number of subsections.
    """
    Nexp = len(Yhist)
    ny = Yhist[0].shape[1]
    nu = Uhist[0].shape[1]
    n = Uhist[0].shape[0]
    YU_hist = np.empty((Nexp, n*(ny+nu)))
    for i in range(Nexp):
        yhist = Yhist[i].reshape(-1)  # (n, ny) -> (n*ny,)
        uhist = Uhist[i].reshape(-1)  # (n, nu) -> (n*nu,)
        yu_hist = np.hstack((yhist, uhist))  # (n*ny+n*nu,)
        YU_hist[i, :] = yu_hist
    return YU_hist


def create_hist_for_test(Yhist: np.ndarray, Uhist: np.ndarray):
    """Concatenates the input and output histories for state reconstruction for the test data.

    Args:
        Yhist (ndarray) : Numpy array with size n-by-ny containing the past output values.
        Uhist (ndarray) : Numpy array with size n-by-nu containing the past input values.

    Returns:
        The concatenated 1D numpy array with size n*(ny+nu) for state reconstruction.
    """
    yhist = vec_reshape(Yhist).reshape(-1)
    uhist = vec_reshape(Uhist).reshape(-1)
    return np.hstack((yhist, uhist))


def create_ndarray_from_list(X: list):
    """Creates a single numpy array from a list of numpy arrays with shapes Ni-by-m, where m can be nz, ny, or nu.

    Args:
        X (list of ndarrays) : List containing the numpy arrays to be concatenated.

    Returns:
        var_array (ndarray) : The concatenated array.
    """
    var_array = np.array([])
    for i in range(len(X)):
        var_i = vec_reshape(X[i])
        if i == 0:
            var_array = var_i
        else:
            var_array = np.vstack((var_array, var_i))
    return var_array


def normalize_data(X: np.ndarray | list, mean=None, std=None):
    """Normalizes a data sequence.

    Args:
        X (ndarray or list of ndarrays) : Data to be normalized.
        mean (ndarray, optional) : The used mean value for standardization. If None, the mean of X is used. (default: None)
        std (ndarray, optional) : The used standard deviation value for standardization. If None, the standard
            deviation of X is used. (default: None)

    Returns:
        X_scaled (ndarray or list of ndarrays) : The normalized data.
        mean (ndarray) : The (computed) mean of the data.
        std (ndarray) : The (computed) standard deviation of the data.
    """
    if isinstance(X, list) and (mean is None or std is None):
        X_array = create_ndarray_from_list(X)
    elif not isinstance(X, list) and (mean is None or std is None):
        X_array = vec_reshape(X).copy()

    if mean is None:
        mean = np.mean(X_array, axis=0)

    if std is None:
        std = np.std(X_array, axis=0)

    if isinstance(X, list):
        X_scaled = X
        for i in range(len(X)):
            X_scaled[i] = (X[i] - mean) / std
    else:
        X_scaled = (X - mean) / std
    return X_scaled, mean, std


def back_scale_data(X: np.ndarray | list, mean: np.ndarray, std: np.ndarray):
    """Back-scales the data sequence with given mean and standard deviation values.

    Args:
        X (ndarray or list of ndarrays) : Data to be normalized.
        mean (ndarray) : The used mean value for de-normalization. If None, the mean of X is used.
        std (ndarray) : The used standard deviation value for de-normalization.

    Returns:
        X_back (ndarray or list of ndarrays) : The back-scaled data sequence.
    """
    if isinstance(X, list):
        X_back = X
        for i in range(len(X)):
            X_back[i] = std * X[i] + mean
    else:
        X_back = std * X + mean
    return X_back


def get_nu_ny_and_auto_norm(Y_train, U_train):

    if isinstance(Y_train, list):
        Y_train_0 = vec_reshape(Y_train[0])
        U_train_0 = vec_reshape(U_train[0])
    else:
        Y_train_0 = vec_reshape(Y_train)
        U_train_0 = vec_reshape(U_train)

    nu = U_train_0.shape[1]
    ny = Y_train_0.shape[1]

    _, y_mean, y_std = normalize_data(Y_train)
    _, u_mean, u_std = normalize_data(U_train)

    norm = dict()
    norm["y_mean"] = y_mean
    norm["y_std"] = y_std
    norm["u_mean"] = u_mean
    norm["u_std"] = u_std

    return nu, ny, norm
