import time
import jaxopt
from functools import partial
from jax_sysid.models import Model, default_small_tau_th, epsil_lasso, l1reg, l2reg, linreg, adam_solver, get_bounds
from jax_sysid.utils import vec_reshape, lbfgs_options
from deepSI_jax.networks import *
from deepSI_jax.data_prep import create_multi_shooting_data, create_multi_shooting_data_with_hist, create_hist_for_test, normalize_data, back_scale_data
from deepSI_jax.utils import xsat
import warnings
import sys


class SUBNET(Model):
    """SUBNET model implementation for identifying discrete-time ANN-SS models based on the jax_sysid.Model class.
    For more details about the methods: https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf
    And about jax-sysid: https://ieeexplore.ieee.org/abstract/document/10882922"""
    def __init__(self, nx: int, ny: int, nu: int, norm=None, f_args={}, h_args={}, use_encoder=True, encoder_lag=10,
                 encoder_args={}, seed=0):
        """ Initializes the model structure.

        Args:
            nx (int) : Model state dimension.
            ny (int) : Output dimension.
            nu (int) : Input dimension.
            norm (dict) : Dictionary containing the mean and standard deviation values for normalization. If None,
                no normalization is applied (i.e., zero-mean and standard deviation of 1 is assumed). (default: None)
            f_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition function.
                Dictionary entries are 'hidden_layers' specifying the number of hidden layers, 'nodes_per_layer' specifying
                the applied number of nodes (neurons) per layer, and 'activation' for the activation function.
            h_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map function.
                    Dictionary entries are the same as for ``f_args``, with the addition of 'feedthrough' (bool) which
                    indicates weather the current output values depends on the current input value (True) or not (False).
            use_encoder (bool) : Whether to use encoder or not. (default: True)
            encoder_lag (int) : Encoder lag for state reconstruction, i.e., the number of past IO data used for
                estimating x0. (default: True)
            encoder_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the encoder.
                Dictionary entries are the same as for ``f_args``.
            seed (int) : Random seed for initialization. (default: 0)
        """

        if norm is None:
            self.norm = dict()
            self.norm['y_mean'] = np.zeros(ny)
            self.norm['y_std'] = np.ones(ny)
            self.norm['u_mean'] = np.zeros(nu)
            self.norm['u_std'] = np.ones(nu)
        else:
            self.norm = norm

        set_default_net_struct_if_necessary(f_args)
        set_default_net_struct_if_necessary(h_args)
        if use_encoder:
            set_default_net_struct_if_necessary(encoder_args)
            self.encoder_lag = encoder_lag
            f_net, h_net, self.encoder_fcn, init_params = gen_f_h_encoder_networks(nx=nx, ny=ny, nu=nu, encoder_lag=encoder_lag,
                                                                                   f_args=f_args, h_args=h_args, encoder_args=encoder_args,
                                                                                   seed=seed)
        else:
            self.encoder_lag = 0
            self.encoder_fcn = None
            f_net, h_net, init_params = gen_f_h_networks(nx=nx, ny=ny, nu=nu, f_args=f_args, h_args=h_args, seed=seed)
        super().__init__(nx=nx, ny=ny, nu=nu, state_fcn=f_net, output_fcn=h_net)
        self.init(params=init_params)
        self.isMultiShooting = False
        self.isEncoderUsed = use_encoder
        self.T = None
        self.T_overlap = 0

    def set_loss_fun(self, T=None, T_overlap=0, output_loss=None, l2_reg=0.0, l1_reg=0.0, group_lasso_reg=0.0, group_lasso_fcn=None,
                     l2_reg_x0=0., zero_coeff=0., custom_regularization=None, xsat=None):
        """Method for setting the loss function parameters.

        Args:
            T (int) : Truncation length. If None, the simulation loss function is used, if T>0, the truncated prediction
                function is applied. (default: None)
            T_overlap (int) : The length of the overlap between each subsection in the truncated prediction loss calculation.
                Only considered if T is not None, otherwise ignored. (default: 0)
            output_loss (function) : Loss function penalizing output fit errors, loss=output_loss(Yhat,Y), where Yhat is
                the sequence of predicted outputs and Y is the measured output. If None, use standard mean squared error
                loss=sum((Yhat-Y)**2)/Y.shape[0]. (default: None)
            l2_reg (float) : Coefficient of the L2 regularization on model parameters. (default: 0.0)
            l1_reg (float) : Coefficient of the L1 regularization on model parameters. (default: 0.0)
            group_lasso_reg (float) : Coefficient of the group lasso regularization for automatic model order selection.
                (default: 0.0)
            group_lasso_fcn (function) : #TODO: update this when implemented the default version
            l2_reg_x0 (float) : Coefficient of the L2 regularization on the initial states. (default: 0.0)
            zero_coeff (float) : Entries smaller than zero_coeff are set to zero. Useful when L1 or group lasso
                regularization is applied. (default: 0.0)
            custom_regularization (function) : Additional custom regularization term, a function of the model parameters
                and initial state, as custom_regularization(params, x0). If None, no additional regularization is applied.
                (default: None)
            xsat (float) : Saturation value for state variables, forced during training to avoid numerical issues.
                If None, no saturation value is applied. (default: None)
        """
        self.T = T
        self.T_overlap = T_overlap
        if self.T is not None and self.T > 0:
            self.isMultiShooting = True
        if self.isEncoderUsed:
            train_x0 = False
        else:
            train_x0 = True
        if not self.isEncoderUsed and l2_reg_x0 > 0:
            rho_x0 = l2_reg_x0
        elif self.isEncoderUsed and l2_reg_x0 > 0:
            rho_x0 = 0.0
            warnings.warn("Regularization of the initial states can not be applied when encoder network is used!")
        else:
            rho_x0 = 0.
        self.loss(output_loss=output_loss, rho_x0=rho_x0, rho_th=l2_reg, tau_th=l1_reg, tau_g=group_lasso_reg, train_x0=train_x0,
                  group_lasso_fcn=group_lasso_fcn, zero_coeff=zero_coeff, custom_regularization=custom_regularization,
                  xsat=xsat)

    def simulate(self, x0: np.ndarray | jnp.ndarray | list, U: np.ndarray | list):
        """Simulates the model on a test data. Automatic normalization and back-scaling is applied.

        Args:
            x0 (ndarray or list of ndarrays) : Initial state to start the simulation, must be (model.nx,) shaped.
                If the model is evaluated on multiple data sequences at the same time, this must be a list, containing
                the estimated initial states.
            U (ndarray ir list of ndarrays) : Input for simulation as an N-by-nu numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-nu numpy arrays.

        Returns:
            Y_back_scaled (ndarray or list of ndarrays) : Back-scaled simulated output for each data sequence.
            X (ndarray or list of ndarrays) : Simulated states for each data sequence.
        """
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])

        @jax.jit
        def model_step(x, u):
            y = jnp.hstack((self.output_fcn(x, u, self.params), x))
            x = self.state_fcn(x, u, self.params).reshape(-1)
            return x, y

        if isinstance(U_norm, list):
            N_meas = len(U_norm)
            Y = []
            X = []
            for i in range(N_meas):
                x = x0[i].copy().reshape(-1)
                u = vec_reshape(U_norm[i])
                _, YX = jax.lax.scan(model_step, x, u)
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        else:
            x = x0.copy().reshape(-1)
            _, YX = jax.lax.scan(model_step, x, vec_reshape(U_norm))
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]

        Y_back_scaled = back_scale_data(Y, self.norm['y_mean'], self.norm['y_std'])
        return Y_back_scaled, X


    def encoder_estim_x0(self, Y: np.ndarray | list, U: np.ndarray | list):
        """Estimates the initial state value with the trained encoder function from past IO data.

        Args:
            Y (ndarray or list of ndarrays) : Output values of the test data with size N-by-ny or list or Ni-by-ny sized
                numpy arrays.
            U (ndarray) : Input values of the test data with size N-by-nu or list or Ni-by-nu sized numpy arrays.

        Returns:
            x0 (ndarray or liost of ndarrays) : The estimated initial state value or batched (list of jax.numpy arrays)
                if multiple experiments are provided.
        """
        Y_norm, _, _ = normalize_data(Y, self.norm['y_mean'], self.norm['y_std'])
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])
        if isinstance(Y_norm, list):
            Nmeas = len(Y_norm)
            x0 = []
            for i in range(Nmeas):
                yu_hist_i = self._check_state_reconstruction_length(Y_norm[i], U_norm[i])
                x0.append(self.encoder_fcn(yu_hist_i, self.params).reshape(-1))
        else:
            yu_hist = self._check_state_reconstruction_length(Y_norm, U_norm)
            x0 = self.encoder_fcn(yu_hist, self.params)
        return x0

    def _check_state_reconstruction_length(self, Y, U):
        """Helper function for encoder_estim_x0. Checks if the provided IO data has the right size for encoder-based
        state estimation."""
        Y = vec_reshape(Y)
        U = vec_reshape(U)
        if Y.shape[0] != U.shape[0]:
            raise ValueError("Y and U must have the same number of rows")
        if Y.shape[0] < self.encoder_lag or U.shape[0] < self.encoder_lag:
            raise ValueError("Provide at least n number of data points for state reconstruction!")
        if Y.shape[0] > self.encoder_lag:
            warnings.warn("Too many data points are provided for state reconstruction, only the last n rows are used.")
        if U.shape[0] > self.encoder_lag:
            warnings.warn("Too many data points are provided for state reconstruction, only the last n rows are used.")
        return create_hist_for_test(Y[-self.encoder_lag:, :], U[-self.encoder_lag:, :])

    def generate_SS_forward_function(self):
        """Generates the SS_forward function that performs a step of the model. For custom SUBNET models, this needs to
        be changed."""
        @jax.jit
        def SS_forward(x, u, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
            """
            y = self.output_fcn(x, u, th)
            x = self.state_fcn(x, u, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x = xsat(x, sat)
            return x, y
        return SS_forward

    def prepare_data_for_training(self, Y_train: np.ndarray | list, U_train: np.ndarray | list):
        """Helper function for the fit method. Performs data normalization and creates multi shooting data if necessary."""
        Y, _, _ = normalize_data(Y_train, self.norm['y_mean'], self.norm['y_std'])
        U, _, _ = normalize_data(U_train, self.norm['u_mean'], self.norm['u_std'])

        if self.isMultiShooting and self.isEncoderUsed:
            Y, U, YU_hist = create_multi_shooting_data_with_hist(Y, U, self.T, self.encoder_lag, self.T_overlap)
        elif self.isMultiShooting and not self.isEncoderUsed:
            Y, U = create_multi_shooting_data(Y, U, self.T, self.T_overlap)
            YU_hist = None
        else:
            Y = Y
            U = U
            YU_hist = None

        if isinstance(U, list):
            Nexp = len(U)  # number of experiments
            if not isinstance(Y, list) or len(Y) != Nexp:
                raise (Exception(
                    "\033[1mPlease provide the same number of input and output traces\033[0m"))
            for i in range(Nexp):
                U[i] = vec_reshape(U[i])
                Y[i] = vec_reshape(Y[i])
        else:
            Nexp = 1
            U = [vec_reshape(U)]
            Y = [vec_reshape(Y)]
        return Y, U, YU_hist, Nexp

    def fit(self, Y_train: np.ndarray | list, U_train: np.ndarray | list):
        """Trains a dynamical model using input-output data.

        Args:
            Y_train (ndarray or list of ndararys) : Outputs of the training data set. Must be N-by-ny numpy array or
                a list or Ni-by-ny numpy arrays, where Ni is the length of the i-th experiment.
            U_train (ndarray or list of ndarrays) : Inputs of the training data set. Must be N-by-nu numpy array or
                a list or Ni-by-nu numpy arrays, where Ni is the length of the i-th experiment.
        """
        nx = self.nx
        if nx < 1:
            raise (
                Exception("\033[1mModel order 'nx' must be greater than zero\033[0m"))

        jax.config.update('jax_platform_name', 'cpu')
        if not jax.config.jax_enable_x64:
            # Enable 64-bit computations
            jax.config.update("jax_enable_x64", True)

        adam_epochs = self.adam_epochs
        lbfgs_epochs = self.lbfgs_epochs

        Y, U, YU_hist, Nexp = self.prepare_data_for_training(Y_train, U_train)

        if self.params is None:
            raise (Exception(
                "\033[1mPlease use the init method to initialize the parameters of the model\033[0m"))

        SS_forward = self.generate_SS_forward_function()

        z = self.params

        tau_th = self.tau_th
        tau_g = self.tau_g

        isL1reg = tau_th > 0
        isGroupLasso = (tau_g > 0) and (self.group_lasso_fcn is not None)
        isCustomReg = self.custom_regularization is not None

        if not isL1reg and isGroupLasso:
            tau_th = default_small_tau_th  # add some small L1-regularization, see Lemma 2

        self.isbounded = (self.params_min is not None) or (self.params_max is not None) or (
                self.train_x0 and ((self.x0_min is not None) or (self.x0_max is not None)))
        if self.isbounded or isL1reg or isGroupLasso:
            # define default bounds, in case they are not provided
            if self.params_min is None:
                self.params_min = list()
                for i in range(len(z)):
                    self.params_min.append(-jnp.ones_like(z[i]) * np.inf)
            if self.params_max is None:
                self.params_max = list()
                for i in range(len(z)):
                    self.params_max.append(jnp.ones_like(z[i]) * np.inf)
            if self.train_x0:
                if self.x0_min is None:
                    self.x0_min = [-jnp.ones_like(self.x0) * np.inf] * Nexp
                if not isinstance(self.x0_min, list):
                    # repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min] * Nexp
                if len(self.x0_min) is not Nexp:
                    # number of experiments has changed, repeat the same initial-state bound on all experiments
                    self.x0_min = [self.x0_min[0]] * Nexp
                if self.x0_max is None:
                    self.x0_max = [jnp.ones_like(self.x0) * np.inf] * Nexp
                if not isinstance(self.x0_max, list):
                    self.x0_max = [self.x0_max] * Nexp
                if len(self.x0_max) is not Nexp:
                    self.x0_max = [self.x0_max[0]] * Nexp

        x0 = self.x0
        if x0 is None:
            x0 = [jnp.zeros(nx)] * Nexp
        elif not isinstance(x0, list):
            x0 = [x0] * Nexp

        def train_model(solver, solver_iters, z, x0, J0):
            """
            Internal function for training the model."""
            if solver_iters == 0:
                return z, x0, J0, 0.

            nth = len(z)
            if solver == "LBFGS" and (isGroupLasso or isL1reg):
                # duplicate params to create positive and negative parts
                z.extend(z)
                for i in range(nth):
                    zi = z[i].copy()
                    # we could also consider bounds here, if present
                    z[i] = jnp.maximum(zi, 0.) + epsil_lasso
                    z[nth + i] = -jnp.minimum(zi, 0.) + epsil_lasso

            nzmx0 = len(z)
            if self.train_x0:
                for i in range(Nexp):
                    # one initial state per experiment
                    z.append(x0[i].reshape(-1))
                # in case of group-Lasso, if state number i is removed from A,B,C then the corresponding x0(i)=0
                # because of L2-regularization on x0.

            # total number of optimization variables
            nvars = sum([zi.size for zi in z])

            @jax.jit
            def simulation_loss(th, x0):
                """Internal function implementing the simulation-based loss function calculations."""
                f = partial(SS_forward, th=th, sat=self.xsat)
                cost = 0.
                for i in range(Nexp):
                    _, Yhat = jax.lax.scan(f, x0[i], U[i])
                    cost += self.output_loss(Yhat, Y[i])
                return cost

            @jax.jit
            def multi_shooting_loss(th, x0):
                """Internal function implementing the truncated prediction-based loss function calculations."""
                f = partial(SS_forward, th=th, sat=self.xsat)

                def run_single_exp(x0i, Ui, Yi):
                    _, Yhati = jax.lax.scan(f, x0i, Ui)
                    loss = self.output_loss(Yhati, Yi)
                    return loss

                # vmap of each experiment to speed up the scan operation
                batched_costs = jax.vmap(run_single_exp)(jnp.stack(x0), jnp.stack(U), jnp.stack(Y))

                return jnp.sum(batched_costs) / Nexp

            @jax.jit
            def multi_shooting_loss_encoder(th, x0):
                """Internal function implementing the truncated prediction-based loss function calculations with
                encoder-based state estimation."""
                def single_state_estim(YU_histi):
                    return self.encoder_fcn(YU_histi, th)

                X0 = jax.vmap(single_state_estim)(YU_hist)

                f = partial(SS_forward, th=th, sat=self.xsat)

                def run_single_exp(x0i, Ui, Yi):
                    _, Yhati = jax.lax.scan(f, x0i, Ui)
                    loss = self.output_loss(Yhati, Yi)
                    return loss

                # vmap of each experiment to speed up the scan operation
                batched_costs = jax.vmap(run_single_exp)(X0, jnp.stack(U), jnp.stack(Y))

                return jnp.sum(batched_costs) / Nexp

            @jax.jit
            def loss(th, x0):
                """Internal function: making a selection for the applied loss function type."""
                if self.isMultiShooting and self.isEncoderUsed:
                    return multi_shooting_loss_encoder(th, x0)
                elif self.isMultiShooting and not self.isEncoderUsed:
                    return multi_shooting_loss(th, x0)
                else:
                    return simulation_loss(th, x0)

            t_solve = time.time()

            if solver == "Adam":
                if self.train_x0:
                    @jax.jit
                    def J(z):
                        th = z[:nzmx0]
                        x0 = z[nzmx0:]
                        cost = loss(th, x0) + self.rho_x0 * \
                               sum([jnp.sum(x0i ** 2)
                                    for x0i in x0]) + self.rho_th * l2reg(th)
                        if isL1reg:
                            cost += tau_th * l1reg(th)
                        if isGroupLasso:
                            cost += tau_g * self.group_lasso_fcn(th, x0)
                        if isCustomReg:
                            cost += self.custom_regularization(th, x0)
                        return cost
                else:
                    @jax.jit
                    def J(z):
                        cost = loss(z, x0) + self.rho_th * l2reg(z)
                        if isL1reg:
                            cost += tau_th * l1reg(z)
                        if isGroupLasso:
                            cost += tau_g * self.group_lasso_fcn(z, x0)
                        if isCustomReg:
                            cost += self.custom_regularization(z, x0)
                        return cost

                def JdJ(z):
                    return jax.value_and_grad(J)(z)

                lb = None
                ub = None
                if self.isbounded:
                    lb = self.params_min
                    ub = self.params_max
                    if self.train_x0:
                        lb.append(self.x0_min)
                        ub.append(self.x0_max)

                z, Jopt = adam_solver(
                    JdJ, z, solver_iters, self.adam_eta, self.iprint, lb, ub)

            elif solver == "LBFGS":
                # L-BFGS-B params (no L1 regularization)
                options = lbfgs_options(
                    min(self.iprint, 90), solver_iters, self.lbfgs_tol, self.memory)

                if self.iprint > -1:
                    print(
                        "Solving NLP with L-BFGS (%d optimization variables) ..." % nvars)

                if isGroupLasso or isL1reg:
                    bounds = get_bounds(
                        z[0:nth], epsil_lasso, self.params_min, self.params_max)
                    if self.train_x0:
                        bounds[0].append(self.x0_min)
                        bounds[1].append(self.x0_max)

                if not isGroupLasso:
                    if not isL1reg:
                        if self.train_x0:
                            @jax.jit
                            def J(z):
                                th = z[:nzmx0]
                                x0 = z[nzmx0:]
                                cost = loss(th, x0) + self.rho_x0 * \
                                       sum([jnp.sum(x0i ** 2)
                                            for x0i in x0]) + self.rho_th * l2reg(th)
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost
                        else:
                            @jax.jit
                            def J(z):
                                cost = loss(z, x0) + self.rho_th * l2reg(z)
                                if isCustomReg:
                                    cost += self.custom_regularization(z, x0)
                                return cost
                        if not self.isbounded:
                            solver = jaxopt.ScipyMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z)
                        else:
                            lb = self.params_min
                            ub = self.params_max
                            if self.train_x0:
                                lb.append(self.x0_min)
                                ub.append(self.x0_max)
                            solver = jaxopt.ScipyBoundedMinimize(
                                fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                            z, state = solver.run(z, bounds=(lb, ub))
                        iter_num = state.iter_num
                        Jopt = state.fun_val
                    else:
                        # Optimize wrt to split positive and negative part of model parameters
                        if self.train_x0:
                            @jax.jit
                            def J(z):
                                x0 = z[nzmx0:]
                                th = [z1 - z2 for (z1, z2)
                                      in zip(z[0:nth], z[nth:2 * nth])]
                                cost = loss(th, x0) + self.rho_x0 * sum(
                                    [jnp.sum(x0i ** 2) for x0i in x0]) + self.rho_th * l2reg(
                                    z[0:nth]) + self.rho_th * l2reg(
                                    z[nth:2 * nth]) + tau_th * linreg(z[0:nth]) + tau_th * linreg(z[nth:2 * nth])
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost
                        else:
                            @jax.jit
                            def J(z):
                                th = [z1 - z2 for (z1, z2)
                                      in zip(z[0:nth], z[nth:2 * nth])]
                                cost = loss(th, x0) + self.rho_th * l2reg(z[0:nth]) + self.rho_th * l2reg(
                                    z[nth:2 * nth]) + tau_th * linreg(z[0:nth]) + tau_th * linreg(z[nth:2 * nth])
                                if isCustomReg:
                                    cost += self.custom_regularization(th, x0)
                                return cost

                        solver = jaxopt.ScipyBoundedMinimize(
                            fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                        z, state = solver.run(z, bounds=bounds)
                        z[0:nth] = [
                            z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                        iter_num = state.iter_num
                        Jopt = state.fun_val

                else:  # group Lasso
                    if self.train_x0:
                        @jax.jit
                        def J(z):
                            x0 = z[nzmx0:]
                            th = [z1 - z2 for (z1, z2)
                                  in zip(z[0:nth], z[nth:2 * nth])]
                            cost = loss(th, x0) + self.rho_x0 * sum(
                                [jnp.sum(x0i ** 2) for x0i in x0]) + self.rho_th * l2reg(
                                z[0:nth]) + self.rho_th * l2reg(
                                z[nth:2 * nth]) + tau_th * linreg(z[0:nth]) + tau_th * linreg(z[nth:2 * nth])
                            if tau_g > 0:
                                cost += tau_g * \
                                        self.group_lasso_fcn(
                                            [z1 + z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])], x0)
                            if isCustomReg:
                                cost += self.custom_regularization(th, x0)
                            return cost
                    else:
                        @jax.jit
                        def J(z):
                            th = [z1 - z2 for (z1, z2)
                                  in zip(z[0:nth], z[nth:2 * nth])]
                            cost = loss(th, x0) + self.rho_th * l2reg(z[0:nth]) + self.rho_th * l2reg(
                                z[nth:2 * nth]) + tau_th * linreg(z[0:nth]) + tau_th * linreg(z[nth:2 * nth])
                            if tau_g > 0:
                                cost += tau_g * \
                                        self.group_lasso_fcn(
                                            [z1 + z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])], x0)
                            if isCustomReg:
                                cost += self.custom_regularization(th, x0)
                            return cost

                    solver = jaxopt.ScipyBoundedMinimize(
                        fun=J, tol=self.lbfgs_tol, method="L-BFGS-B", maxiter=solver_iters, options=options)
                    z, state = solver.run(z, bounds=bounds)
                    z[0:nth] = [
                        z1 - z2 for (z1, z2) in zip(z[0:nth], z[nth:2 * nth])]
                    iter_num = state.iter_num
                    Jopt = state.fun_val

                print('L-BFGS-B done in %d iterations.' % iter_num)

            else:
                raise (Exception("\033[1mUnknown solver\033[0m"))

            if self.train_x0:
                x0 = [z[nzmx0 + i].reshape(-1) for i in range(Nexp)]

            t_solve = time.time() - t_solve
            return z[0:nth], x0, Jopt, t_solve

        z, x0, Jopt, t_solve1 = train_model('Adam', adam_epochs, z, x0, np.inf)
        z, x0, Jopt, t_solve2 = train_model('LBFGS', lbfgs_epochs, z, x0, Jopt)
        t_solve = t_solve1 + t_solve2

        x0 = [np.array(x0i) for x0i in x0]

        if not self.isbounded:
            # reset original bounds, possibly altered in case of L1-regularization or group-Lasso and L-BFGS is used
            self.params_min = None
            self.params_max = None
            self.x0_min = None
            self.x0_max = None

        # Zero coefficients smaller than zero_coeff in absolute value
        for i in range(len(z)):
            z[i] = np.array(z[i])
            z[i][np.abs(z[i]) <= self.zero_coeff] = 0.

        self.params = z

        # Zero coefficients smaller than zero_coeff in absolute value
        for i in range(Nexp):
            x0[i][np.abs(x0[i]) <= self.zero_coeff] = 0.0

        # Check if state saturation is active.
        if self.xsat is not None:
            for i in range(Nexp):
                # this overrides possible predict() methods defined in subclasses
                _, Xt = Model.predict(self, x0[i], U[i], 0, 0)
                sat_activated = np.any(Xt > self.xsat) or np.any(Xt < -self.xsat)
                if sat_activated:
                    print(
                        "\033[1mWarning: state saturation is active at the solution. \nYou may have to increase the values of 'xsat' or 'rho_x0'\033[0m")
        else:
            sat_activated = None

        # Check model sparsity
        sparsity = dict()
        sparsity["nonzero_parameters"] = [np.sum([np.sum(np.abs(z[i]) > self.zero_coeff) for i in range(
            len(z))]), np.sum([z[i].size for i in range(len(z))])]

        self.x0 = x0
        if Nexp == 1:
            self.x0 = self.x0[0]
        self.Jopt = Jopt
        self.t_solve = t_solve
        self.Nexp = Nexp
        self.sat_activated = sat_activated
        self.sparsity = sparsity
        return

    def learn_x0(self, U: np.ndarray | list, Y: np.ndarray | list, rho_x0=None, RTS_epochs=1, verbosity=True,
                 LBFGS_refinement=True, LBFGS_rho_x0=0., lbfgs_epochs=1000, Q=None, R=None):
        """Estimate x0 by L-BFGS optimization, and providing an initial guess by Rauch–Tung–Striebel smoothing.

        Args:
            U (ndarray or list of ndararys) : Input data. Must be N-by-nu numpy array or a list of Ni-by-nu numpy arrays.
            Y (ndarray or list of ndarrays) : Output data. Must be N-by-ny numpy array or a list of Ni-by-ny numpy arrays.
            rho_x0 (float) : L2 regularization for the initial state. Only used by the EKF-based guessing. If None, or
                zero, 1e-4 is used. (default: None)
            RTS_epochs (int) : Number of forward EKF and backward RTS passes. (default: 1)
            verbosity (bool) : If false, removes printout of operations. (default: True)
            LBFGS_refinement (bool) : If True, refine the RTS solution via L-BFGS optimization. (default: True)
            LBFGS_rho_x0 (float) : L2-regularization used by L-BFGS. (default: 0.)
            lbfgs_epochs (int) : Max number of L-BFGS iterations. (default: 1000)
            Q (ndarray) : Process noise covariance matrix. If None, 1.e-5*I is applied. (default: None)
            R (ndarray) : Measurement noise covariance matrix. If None, the identity matrix I is applied. (default: None)

        Returns:
            x0 (ndarary or list of ndararys) : Estimated initial states.
        """
        if self.rho_x0 == 0. and (rho_x0 is None or rho_x0 <= 0.):
            rho_x0 = 1e-4  # learn_x0 method uses rho_x0 to initialize covariance matrix, so it should be larger than zero
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])
        Y_norm, _, _ = normalize_data(Y, self.norm['y_mean'], self.norm['y_std'])

        if isinstance(U_norm, list):
            Nexp = len(U_norm)
            x0 = []
            for i in range(Nexp):
                x0.append(super().learn_x0(vec_reshape(U_norm[i]), vec_reshape(Y_norm[i]), rho_x0, RTS_epochs, verbosity,
                                           LBFGS_refinement, LBFGS_rho_x0, lbfgs_epochs, Q, R))
        else:
            x0 = super().learn_x0(vec_reshape(U_norm), vec_reshape(Y_norm), rho_x0, RTS_epochs, verbosity, LBFGS_refinement,
                                  LBFGS_rho_x0, lbfgs_epochs, Q, R)
        return x0


class SUBNET_innovation(SUBNET):
    """SUBNET model implementation with innovation noise filter for identifying discrete-time ANN-SS models with
    process noise structure. For more information: https://www.sciencedirect.com/science/article/pii/S0005109823003710"""
    def __init__(self, nx: int, ny: int, nu: int, norm=None, f_args={}, h_args={}, use_encoder=True, encoder_lag=10,
                 encoder_args={}, seed=0):
        """ Initializes the model structure.

        Args:
            nx (int) : Model state dimension.
            ny (int) : Output dimension.
            nu (int) : Input dimension.
            norm (dict) : Dictionary containing the mean and standard deviation values for normalization. If None,
                no normalization is applied (i.e., zero-mean and standard deviation of 1 is assumed). (default: None)
            f_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition function.
                Dictionary entries are 'hidden_layers' specifying the number of hidden layers, 'nodes_per_layer' specifying
                the applied number of nodes (neurons) per layer, and 'activation' for the activation function.
            h_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map function.
                    Dictionary entries are the same as for ``f_args``, with the addition of 'feedthrough' (bool) which
                    indicates weather the current output values depends on the current input value (True) or not (False).
            use_encoder (bool) : Whether to use encoder or not. (default: True)
            encoder_lag (int) : Encoder lag for state reconstruction, i.e., the number of past IO data used for
                estimating x0. (default: True)
            encoder_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the encoder.
                Dictionary entries are the same as for ``f_args``.
            seed (int) : Random seed for initialization. (default: 0)
        """
        super().__init__(nx=nx, ny=ny, nu=nu, norm=norm, f_args=f_args, h_args=h_args, use_encoder=use_encoder,
                         encoder_lag=encoder_lag, encoder_args=encoder_args, seed=seed)

        # re-initialize f-net with correct dimensions
        set_default_net_struct_if_necessary(f_args)
        set_default_net_struct_if_necessary(h_args)
        if use_encoder:
            set_default_net_struct_if_necessary(encoder_args)
            self.encoder_lag = encoder_lag
            f_net, _, _, init_params = gen_f_h_encoder_networks(nx=nx, ny=ny, nu=nu, encoder_lag=encoder_lag, f_args=f_args,
                                                                h_args=h_args, encoder_args=encoder_args, seed=seed,
                                                                innovation_noise_struct=True)
        else:
            self.encoder_lag = 0
            self.encoder_fcn = None
            f_net, _, init_params = gen_f_h_networks(nx=nx, ny=ny, nu=nu, f_args=f_args, h_args=h_args, seed=seed,
                                                     innovation_noise_struct=True)

        self.state_fcn = f_net
        self.init(params=init_params)

    def simulate(self, x0: np.ndarray | jnp.ndarray | list, U: np.ndarray | list):
        """Simulates the model on a test data. Automatic normalization and back-scaling is applied. To enable the forward
        pass of the innovation noise filter, theprediction error is assumed to be zero.

        Args:
            x0 (ndarray or list of ndarrays) : Initial state to start the simulates, must be (model.nx,) shaped.
                If the model is evaluated on multiple data sequences at the same time, this must be a list, containing
                the estimated initial states.
            U (ndarray ir list of ndarrays) : Input for simulation as an N-by-nu numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-nu numpy arrays.

        Returns:
            Y_back_scaled (ndarray or list of ndarrays) : Back-scaled simulated output for each data sequence.
            X (ndarray or list of ndarrays) : Simulated states for each data sequence.
        """
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])

        @jax.jit
        def model_step(x, u):
            y = jnp.hstack((self.output_fcn(x, u, self.params), x))
            e = np.zeros(self.ny)  # for simulation, we can not use real y values --> estimate the output noise as zero
            x = self.state_fcn(x, u, e, self.params).reshape(-1)
            return x, y

        if isinstance(U_norm, list):
            N_meas = len(U_norm)
            Y = []
            X = []
            for i in range(N_meas):
                x = x0[i].copy().reshape(-1)
                u = vec_reshape(U_norm[i])
                _, YX = jax.lax.scan(model_step, x, u)
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        else:
            x = x0.copy().reshape(-1)
            _, YX = jax.lax.scan(model_step, x, vec_reshape(U_norm))
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]

        Y_back_scaled = back_scale_data(Y, self.norm['y_mean'], self.norm['y_std'])
        return Y_back_scaled, X

    def predict_one_step_ahead(self, x0: np.ndarray | jnp.ndarray | list, U: np.ndarray | list, Y: np.ndarray | list):
        """Predicts the model output on a test data in a one-ste-ahead manner. Automatic normalization and back-scaling
        is applied. The current output values are only utilized for predicting the next state values with the innovation
        noise structure.

        Args:
            x0 (ndarray or list of ndarrays) : Initial state to start the predictions, must be (model.nx,) shaped.
                If the model is evaluated on multiple data sequences at the same time, this must be a list, containing
                the estimated initial states.
            U (ndarray ir list of ndarrays) : Input for predictions as an N-by-nu numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-nu numpy arrays.
            Y (ndarray or list of ndarrays) : Measured output values as an N-by-ny numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-ny numpy arrays.

        Returns:
            Y_back_scaled (ndarray or list of ndarrays) : Back-scaled simulated output for each data sequence.
            X (ndarray or list of ndarrays) : Simulated states for each data sequence.
        """
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])
        Y_norm, _, _ = normalize_data(Y, self.norm['y_mean'], self.norm['y_std'])

        @jax.jit
        def model_step(x, uy):
            u = uy[:self.nu]
            y_true = uy[self.nu:]
            yhat = self.output_fcn(x, u, self.params)
            y = jnp.hstack((yhat, x))
            e = y_true - yhat  # for prediction, we can use the real y values
            x = self.state_fcn(x, u, e, self.params).reshape(-1)
            return x, y

        if isinstance(U_norm, list):
            N_meas = len(U_norm)
            Y = []
            X = []
            for i in range(N_meas):
                x = x0[i].copy().reshape(-1)
                u = vec_reshape(U_norm[i])
                y = vec_reshape(Y_norm[i])
                uy = np.hstack((u, y))
                _, YX = jax.lax.scan(model_step, x, uy)
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        else:
            x = x0.copy().reshape(-1)
            UY = np.hstack((vec_reshape(U_norm), vec_reshape(Y_norm)))
            _, YX = jax.lax.scan(model_step, x, UY)
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]

        Y_back_scaled = back_scale_data(Y, self.norm['y_mean'], self.norm['y_std'])
        return Y_back_scaled, X

    def generate_SS_forward_function(self):
        """Generates the SS_forward function that performs a step of the model. The current output error value is used
        for the state transition."""
        @jax.jit
        def SS_forward(x, uy, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
            """
            # uy contains both the input and output values: uy = [u, y]
            u = uy[:self.nu]
            y_true = uy[self.nu:]
            yhat = self.output_fcn(x, u, th)
            e = y_true - yhat  # estimate the output noise
            x = self.state_fcn(x, u, e, th).reshape(-1)

            # saturate states to avoid numerical issues due to instability
            x = xsat(x, sat)
            return x, yhat
        return SS_forward

    def prepare_data_for_training(self, Y_train: np.ndarray | list, U_train: np.ndarray | list):
        """Helper function for the fit method. Performs data normalization and creates multi shooting data if necessary.
        The returned U array contains the input and output values as well, since now the output values are utilized for
        model evaluations."""
        Y, _, _ = normalize_data(Y_train, self.norm['y_mean'], self.norm['y_std'])
        U, _, _ = normalize_data(U_train, self.norm['u_mean'], self.norm['u_std'])

        if self.isMultiShooting and self.isEncoderUsed:
            Y, U, YU_hist = create_multi_shooting_data_with_hist(Y, U, self.T, self.encoder_lag, self.T_overlap)
        elif self.isMultiShooting and not self.isEncoderUsed:
            Y, U = create_multi_shooting_data(Y, U, self.T, self.T_overlap)
            YU_hist = None
        else:
            Y = Y
            U = U
            YU_hist = None

        # stack U to contain both the control inputs and the output values (as y is necessary to estimate the output noise)
        if isinstance(U, list):
            Nexp = len(U)  # number of experiments
            if not isinstance(Y, list) or len(Y) != Nexp:
                raise (Exception(
                    "\033[1mPlease provide the same number of input and output traces\033[0m"))
            for i in range(Nexp):
                U[i] = vec_reshape(U[i])
                Y[i] = vec_reshape(Y[i])
                U[i] = np.hstack((U[i], Y[i]))
        else:
            Nexp = 1
            Ui = vec_reshape(U)
            Yi = vec_reshape(Y)
            U = [np.hstack((Ui, Yi))]
            Y = [Yi]
        return Y, U, YU_hist, Nexp

    def learn_x0_single(self, U: np.ndarray, Y: np.ndarray, rho_x0=None, RTS_epochs=1, verbosity=True,
                        LBFGS_refinement=False, LBFGS_rho_x0=0., lbfgs_epochs=1000, Q=None, R=None):
        """Helper function for learn_xo. Estimates x0 for a single measurement record by L-BFGS optimization, and
        provides an initial guess by Rauch–Tung–Striebel smoothing. The EKF and RTS part only considers the deterministic
        part of the model, while the L-BFGS-based optimization takes the full model into account with the innovation noise
        structure.
        """
        nx = self.nx
        ny = self.ny
        N = U.shape[0]

        @jax.jit
        def Ck(x, u):
            return jax.jacrev(self.output_fcn)(x, u=u, params=self.params)

        # EKF + RTS smoothing should only consider the deterministic part of f
        def state_fcn_deterministic(x, u, params):
            return self.state_fcn(x, u, jnp.zeros(ny), params)

        @jax.jit
        def Ak(x, u):
            return jax.jacrev(state_fcn_deterministic)(x, u=u, params=self.params)

        if rho_x0 is None:
            rho_x0 = self.rho_x0
        if R is None:
            R = np.eye(ny)
        if Q is None:
            Q = 1.e-5 * np.eye(nx)

        # Forward EKF pass:
        @jax.jit
        def EKF_update(state, yuk):
            x, P, mse_loss = state
            yk = yuk[:ny]
            u = yuk[ny:]

            # measurement update
            y = self.output_fcn(x, u, self.params)
            Ckk = Ck(x, u)
            PC = P @ Ckk.T
            # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
            # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
            M = jax.scipy.linalg.solve((R+Ckk@PC), PC.T, assume_a='pos').T
            e = yk-y
            mse_loss += np.sum(e**2)  # just for monitoring purposes
            x1 = x + M@e  # x(k | k)

            # Standard Kalman measurement update
            # P -= M@PC.T
            # P = (P + P.T)/2. # P(k|k)

            # Joseph stabilized covariance update
            IKH = -M@Ckk
            IKH += jnp.eye(nx)
            P1 = IKH@P@IKH.T+M@R@M.T  # P(k|k)

            # Time update
            Akk = Ak(x1, u)
            P2 = Akk@P1@Akk.T+Q
            # P2 = (P2+P2.T)/2.
            x2 = state_fcn_deterministic(x1, u, self.params)
            output = (x1, P1, x2, P2, Akk)

            return (x2, P2, mse_loss), output

        @jax.jit
        def RTS_update(state, input):
            x, P = state
            P1, P2, x1, x2, A = input

            # G=(PP1[k]@AA[k].T)/PP2[k]
            try:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='pos').T
            except:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='gen').T
            x = x1+G@(x-x2)
            P = P1+G@(P-P2)@G.T
            return (x, P), None

        # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
        P = np.eye(nx) / (rho_x0 * N)
        x = np.zeros(nx)

        for epoch in range(RTS_epochs):
            mse_loss = 0.

            # Forward EKF pass
            state = (x, P, mse_loss)
            state, output = jax.lax.scan(EKF_update, state, np.hstack((Y, U)))
            XX1, PP1, XX2, PP2, AA = output
            # PP1 = P(k | k)
            # PP2 = P(k + 1 | k)
            # XX1 = x(k | k)
            # XX2 = x(k + 1 | k)
            mse_loss = state[2]/N

            # RTS smoother pass:
            x = XX2[N-1]
            P = PP2[N-1]
            state = (x, P)
            input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1], AA[::-1])
            state, _ = jax.lax.scan(RTS_update, state, input)
            x, P = state

            if verbosity:
                sys.stdout.write('\033[F')
                print(
                    f"\nRTS smoothing, epoch: {epoch+1: 3d}/{RTS_epochs: 3d}, MSE loss = {mse_loss: 8.6f}")

        x = np.array(x)

        isstatebounded = self.x0_min is not None or self.x0_max is not None
        if isstatebounded:
            lb = self.x0_min
            if isinstance(lb, list):
                lb = lb[0]
            if lb is None:
                lb = -np.inf*np.ones(nx)
            ub = self.x0_max
            if isinstance(ub, list):
                ub = ub[0]
            if ub is None:
                ub = np.inf*np.ones(nx)
            if np.any(x < lb) or np.any(x > ub):
                LBFGS_refinement = True

        if LBFGS_refinement:
            # Refine via L-BFGS with very small penalty on x0
            options = lbfgs_options(
                iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-10, memory=100)

            # now we should consider innovation noise structure
            UY = np.hstack((U, Y))
            @jax.jit
            def SS_step(x, uy):
                u = uy[:self.nu]
                y = uy[self.nu:]
                yhat = self.output_fcn(x, u, self.params)
                e = y - yhat
                x = self.state_fcn(x, u, e, self.params).reshape(-1)
                return x, yhat

            @jax.jit
            def J(x0):
                _, Yhat = jax.lax.scan(SS_step, x0, UY)
                return jnp.sum((Yhat - Y) ** 2) / U.shape[0]+.5*LBFGS_rho_x0*jnp.sum(x0**2)
            if not isstatebounded:
                solver = jaxopt.ScipyMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x)
            else:
                solver = jaxopt.ScipyBoundedMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x, bounds=(lb, ub))
            x = np.array(x)

            if verbosity:
                mse_loss = state.fun_val-.5*LBFGS_rho_x0*np.sum(x**2)
                print(
                    f"\nFinal loss MSE (after LBFGS refinement) = {mse_loss: 8.6f}")
        return x

    def learn_x0(self, U: np.ndarray | list, Y: np.ndarray | list, rho_x0=None, RTS_epochs=1, verbosity=True,
                 LBFGS_refinement=True, LBFGS_rho_x0=0., lbfgs_epochs=1000, Q=None, R=None):
        """Estimate x0 by L-BFGS optimization, and providing an initial guess by Rauch–Tung–Striebel smoothing.

        Args:
            U (ndarray or list of ndararys) : Input data. Must be N-by-nu numpy array or a list of Ni-by-nu numpy arrays.
            Y (ndarray or list of ndarrays) : Output data. Must be N-by-ny numpy array or a list of Ni-by-ny numpy arrays.
            rho_x0 (float) : L2 regularization for the initial state. Only used by the EKF-based guessing. If None, or
                zero, 1e-4 is used. (default: None)
            RTS_epochs (int) : Number of forward EKF and backward RTS passes. (default: 1)
            verbosity (bool) : If false, removes printout of operations. (default: True)
            LBFGS_refinement (bool) : If True, refine the RTS solution via L-BFGS optimization. (default: True)
            LBFGS_rho_x0 (float) : L2-regularization used by L-BFGS. (default: 0.)
            lbfgs_epochs (int) : Max number of L-BFGS iterations. (default: 1000)
            Q (ndarray) : Process noise covariance matrix. If None, 1.e-5*I is applied. (default: None)
            R (ndarray) : Measurement noise covariance matrix. If None, the identity matrix I is applied. (default: None)

        Returns:
            x0 (ndarary or list of ndararys) : Estimated initial states.
        """
        if self.rho_x0 == 0. and (rho_x0 is None or rho_x0 <= 0.):
            rho_x0 = 1e-4  # learn_x0 method uses rho_x0 to initialize covariance matrix, so it should be larger than zero
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])
        Y_norm, _, _ = normalize_data(Y, self.norm['y_mean'], self.norm['y_std'])

        if isinstance(U_norm, list):
            Nexp = len(U_norm)
            x0 = []
            for i in range(Nexp):
                x0.append(self.learn_x0_single(vec_reshape(U_norm[i]), vec_reshape(Y_norm[i]), rho_x0, RTS_epochs,
                                               verbosity, LBFGS_refinement, LBFGS_rho_x0, lbfgs_epochs, Q, R))
        else:
            x0 = self.learn_x0_single(vec_reshape(U_norm), vec_reshape(Y_norm), rho_x0, RTS_epochs, verbosity,
                                      LBFGS_refinement, LBFGS_rho_x0, lbfgs_epochs, Q, R)
        return x0


class SUBNET_separated_noise_model(SUBNET_innovation):
    """SUBNET model implementation with separate process and noise model parametrization for identifying discrete-time
    ANN-SS models with process  noise structures. For more information about the method: https://arxiv.org/pdf/2504.11982"""
    def __init__(self, nx: int, nz: int, ny: int, nu: int, norm=None, fx_args={}, hx_args={}, fz_args={}, hz_args={},
                 use_encoder=True, encoder_lag=10, encoder_args={}, seed=0, warm_start_params=None, freeze_plant_model=False):
        """ Initializes the model structure.

        Args:
            nx (int) : Model state dimension.
            nz (int) : Dimension of the inverse noise process state.
            ny (int) : Output dimension.
            nu (int) : Input dimension.
            norm (dict) : Dictionary containing the mean and standard deviation values for normalization. If None,
                no normalization is applied (i.e., zero-mean and standard deviation of 1 is assumed). (default: None)
            fx_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition
                function of the process model. Dictionary entries are 'hidden_layers' specifying the number of hidden
                layers, 'nodes_per_layer' specifying the applied number of nodes (neurons) per layer, and 'activation'
                for the activation function.
            hx_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map function
                for the process model. Dictionary entries are the same as for ``f_args``, with the addition of
                'feedthrough' (bool) which indicates weather the current output values depends on the current input value
                (True) or not (False).
            fz_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the state transition of
                the inverse noise model. Dictionary entries are the same as for `fx_args`.
            hz_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the output map of
                the inverse noise model. Dictionary entries are the same as for `hx_args`.
            use_encoder (bool) : Whether to use encoder or not. (default: True)
            encoder_lag (int) : Encoder lag for state reconstruction, i.e., the number of past IO data used for
                estimating x0. (default: True)
            encoder_args (dict) : Dictionary containing the hyperparameters of the ANN structure for the encoder.
                Dictionary entries are the same as for ``f_args``.
            seed (int) : Random seed for initialization. (default: 0)
            warm_start_params (list of ndarrays) : If not None, these parameters are used for initializing the process
                part. (default: None)
            freeze_plant_model (bool) : If True and warm_start_params is not None, then the process part is frozen and
                only the noise model (and encoder model)is trained. (default: False)
        """
        super().__init__(nx=nx, ny=ny, nu=nu, norm=norm, use_encoder=use_encoder, encoder_lag=encoder_lag)

        self.nx = nx + nz
        self.nx_x = nx
        self.nx_z = nz
        set_default_net_struct_if_necessary(fx_args)
        set_default_net_struct_if_necessary(hx_args)
        set_default_net_struct_if_necessary(fz_args)
        set_default_net_struct_if_necessary(hz_args)

        if use_encoder and not freeze_plant_model:
            set_default_net_struct_if_necessary(encoder_args)
            fx_net, hx_net, fz_net, hz_net, encoder_net, init_params = gen_fx_hx_fz_hz_encoder_networks(nx=nx, nz=nz, nu=nu, ny=ny,
                                                                                          encoder_lag=encoder_lag,
                                                                                          fx_args=fx_args,
                                                                                          hx_args=hx_args,
                                                                                          fz_args=fz_args,
                                                                                          hz_args=hz_args,
                                                                                          encoder_args=encoder_args,
                                                                                          seed=seed)
        elif not use_encoder and not freeze_plant_model:
            encoder_net = None
            fx_net, hx_net, fz_net, hz_net, init_params = gen_fx_hx_fz_hz_networks(nx=nx, nz=nz, nu=nu, ny=ny, fx_args=fx_args,
                                                                                   hx_args=hx_args, fz_args=fz_args,
                                                                                   hz_args=hz_args, seed=seed)
        elif use_encoder and freeze_plant_model:
            set_default_net_struct_if_necessary(encoder_args)
            fx_net, hx_net, fz_net, hz_net, encoder_net, init_params = gen_fx_hx_fz_hz_encoder_networks(nx=nx, nz=nz, nu=nu, ny=ny,
                                                                                          encoder_lag=encoder_lag,
                                                                                          fx_args=fx_args,
                                                                                          hx_args=hx_args,
                                                                                          fz_args=fz_args,
                                                                                          hz_args=hz_args,
                                                                                          encoder_args=encoder_args,
                                                                                          seed=seed)
        else:   # no encoder and frozen plant parameters
            encoder_net = None
            fx_net, hx_net, fz_net, hz_net, init_params = gen_separate_fx_hx_fz_hz_networks(nx=nx, nz=nz, nu=nu, ny=ny,
                                                                                            fx_args=fx_args, hx_args=hx_args,
                                                                                            fz_args=fz_args, hz_args=hz_args,
                                                                                            seed=seed)

        if warm_start_params is not None and not freeze_plant_model:
            # co-estimate plant+noise model, only use plant model parameters for warm-starting the optimization
            deterministic_part_idx_end = 2 * fx_args['hidden_layers'] + 3 + 2 * hx_args['hidden_layers'] + 2
            for i in range(deterministic_part_idx_end+1):
                init_params[i] = warm_start_params[i]

        if warm_start_params is not None and use_encoder:
            # initialize encoder net to reconstruct x state as the pre-trained model and z state starts from random init.
            init_params[-1] = init_params[-1].at[:self.nx_x, :].set(warm_start_params[-1])  # encoder residual weight
            init_params[-2] = init_params[-2].at[:self.nx_x].set(warm_start_params[-2])  # encoder last layer's bias vector
            init_params[-3] = init_params[-3].at[:self.nx_x, :].set(warm_start_params[-3])  # encoder last layer's weight matrix

        self.encoder_fcn = encoder_net
        if freeze_plant_model:
            @jax.jit
            def plant_f(x, u, params):
                return fx_net(x, u, warm_start_params)

            @jax.jit
            def plant_h(x, u, params):
                return hx_net(x, u, warm_start_params)

            self.state_fcn = plant_f
            self.output_fcn = plant_h
        else:
            self.state_fcn = fx_net
            self.output_fcn = hx_net
        self.noise_state_fcn = fz_net
        self.noise_output_fcn = hz_net
        self.init(init_params)

    def generate_SS_forward_function(self):
        """Generates the SS_forward function that performs a step of the model. With both the process and noise part."""
        @jax.jit
        def SS_forward(xz, uy, th, sat):
            """
            Perform a forward pass of the nonlinear model. States are saturated to avoid possible explosion of state values in case the system is unstable.
            """
            # xz contains both the deterministic and the stochastic states: xz = [x, z]
            x = xz[:self.nx_x]
            z = xz[self.nx_x:]

            # uy contains both the input and output values: uy = [u, y]
            u = uy[:self.nu]
            y_true = uy[self.nu:]

            # compute deterministic model state + output equations
            x_plus = self.state_fcn(x, u, th)
            y_det = self.output_fcn(x, u, th)

            # compute latent state transition
            v = y_true - y_det
            z_plus = self.noise_state_fcn(z, x, u, v, th)

            # add estimated noise value to the deterministic output
            yhat = y_det - self.noise_output_fcn(z, x, u, th)

            # combined states
            xz_plus = jnp.hstack((x_plus, z_plus))
            # saturate states to avoid numerical issues due to instability
            xz_plus = xsat(xz_plus, sat)
            return xz_plus, yhat

        return SS_forward

    def learn_x0_single(self, U: np.ndarray, Y: np.ndarray, rho_x0=None, RTS_epochs=1, verbosity=True,
                        LBFGS_refinement=True, LBFGS_rho_x0=0., lbfgs_epochs=1000, Q=None, R=None):
        """Helper function for learn_xo. Estimates x0 for a single measurement record by L-BFGS optimization, and
        provides an initial guess by Rauch–Tung–Striebel smoothing. The EKF and RTS part only considers the deterministic
        part of the model, while the L-BFGS-based optimization takes the full model into account with the innovation noise
        structure.
        """
        nx = self.nx
        ny = self.ny
        N = U.shape[0]

        def combined_state_transition_fun(xz, u, y, params):
            x = xz[:self.nx_x]
            z = xz[self.nx_x:]

            x_plus = self.state_fcn(x, u, params)
            yhat = self.output_fcn(x, u, params)
            z_plus = self.noise_state_fcn(z, x, u, y-yhat, params)
            return jnp.hstack((x_plus, z_plus))

        def combined_output_fun(xz, u, params):
            x = xz[:self.nx_x]
            z = xz[self.nx_x:]
            return self.output_fcn(x, u, params) - self.noise_output_fcn(z, x, u, params)

        @jax.jit
        def Ck(xz, u):
            return jax.jacrev(combined_output_fun)(xz, u=u, params=self.params)

        @jax.jit
        def Ak(xz, u, y):
            return jax.jacrev(combined_state_transition_fun)(xz, u=u, y=y, params=self.params)

        if rho_x0 is None:
            rho_x0 = self.rho_x0
        if R is None:
            R = np.eye(ny)
        if Q is None:
            Q = 1.e-5 * np.eye(nx)

        # Forward EKF pass:
        @jax.jit
        def EKF_update(state, yuk):
            x, P, mse_loss = state
            yk = yuk[:ny]
            u = yuk[ny:]

            # measurement update
            y = combined_output_fun(x, u, self.params)
            Ckk = Ck(x, u)
            PC = P @ Ckk.T
            # M = PC / (R + C @PC) # this solves the linear system M*(R + C @PC) = PC
            # Note: Matlab's mrdivide A / B = (B'\A')' = np.linalg.solve(B.conj().T, A.conj().T).conj().T
            M = jax.scipy.linalg.solve((R+Ckk@PC), PC.T, assume_a='pos').T
            e = yk-y
            mse_loss += np.sum(e**2)  # just for monitoring purposes
            x1 = x + M@e  # x(k | k)

            # Standard Kalman measurement update
            # P -= M@PC.T
            # P = (P + P.T)/2. # P(k|k)

            # Joseph stabilized covariance update
            IKH = -M@Ckk
            IKH += jnp.eye(nx)
            P1 = IKH@P@IKH.T+M@R@M.T  # P(k|k)

            # Time update
            Akk = Ak(x1, u, yk)
            P2 = Akk@P1@Akk.T+Q
            # P2 = (P2+P2.T)/2.
            x2 = combined_state_transition_fun(x1, u, yk, self.params)
            output = (x1, P1, x2, P2, Akk)

            return (x2, P2, mse_loss), output

        @jax.jit
        def RTS_update(state, input):
            x, P = state
            P1, P2, x1, x2, A = input

            # G=(PP1[k]@AA[k].T)/PP2[k]
            try:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='pos').T
            except:
                G = jax.scipy.linalg.solve(P2, (P1@A.T).T, assume_a='gen').T
            x = x1+G@(x-x2)
            P = P1+G@(P-P2)@G.T
            return (x, P), None

        # L2-regularization on initial state x0, 0.5*rho_x0*||x0||_2^2
        P = np.eye(nx) / (rho_x0 * N)
        x = np.zeros(nx)

        for epoch in range(RTS_epochs):
            mse_loss = 0.

            # Forward EKF pass
            state = (x, P, mse_loss)
            state, output = jax.lax.scan(EKF_update, state, np.hstack((Y, U)))
            XX1, PP1, XX2, PP2, AA = output
            # PP1 = P(k | k)
            # PP2 = P(k + 1 | k)
            # XX1 = x(k | k)
            # XX2 = x(k + 1 | k)
            mse_loss = state[2]/N

            # RTS smoother pass:
            x = XX2[N-1]
            P = PP2[N-1]
            state = (x, P)
            input = (PP1[::-1], PP2[::-1], XX1[::-1], XX2[::-1], AA[::-1])
            state, _ = jax.lax.scan(RTS_update, state, input)
            x, P = state

            if verbosity:
                sys.stdout.write('\033[F')
                print(
                    f"\nRTS smoothing, epoch: {epoch+1: 3d}/{RTS_epochs: 3d}, MSE loss = {mse_loss: 8.6f}")

        x = np.array(x)

        isstatebounded = self.x0_min is not None or self.x0_max is not None
        if isstatebounded:
            lb = self.x0_min
            if isinstance(lb, list):
                lb = lb[0]
            if lb is None:
                lb = -np.inf*np.ones(nx)
            ub = self.x0_max
            if isinstance(ub, list):
                ub = ub[0]
            if ub is None:
                ub = np.inf*np.ones(nx)
            if np.any(x < lb) or np.any(x > ub):
                LBFGS_refinement = True

        if LBFGS_refinement:
            # Refine via L-BFGS with very small penalty on x0
            options = lbfgs_options(
                iprint=-1, iters=lbfgs_epochs, lbfgs_tol=1.e-10, memory=100)

            # now we should consider innovation noise structure
            UY = np.hstack((U, Y))
            @jax.jit
            def SS_step(xz, uy):
                x = xz[:self.nx_x]
                z = xz[self.nx_x:]
                u = uy[:self.nu]
                y = uy[self.nu:]
                x_plus = self.state_fcn(x, u, self.params)
                yhat = self.output_fcn(x, u, self.params)
                z_plus = self.noise_state_fcn(z, x, u, y-yhat, self.params)
                yhat = yhat - self.noise_output_fcn(z, x, u, self.params)
                xz = jnp.hstack((x_plus, z_plus))
                return xz, yhat

            @jax.jit
            def J(x0):
                _, Yhat = jax.lax.scan(SS_step, x0, UY)
                return jnp.sum((Yhat - Y) ** 2) / U.shape[0]+.5*LBFGS_rho_x0*jnp.sum(x0**2)
            if not isstatebounded:
                solver = jaxopt.ScipyMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x)
            else:
                solver = jaxopt.ScipyBoundedMinimize(
                    fun=J, tol=options["ftol"], method="L-BFGS-B", maxiter=options["maxfun"], options=options)
                x, state = solver.run(x, bounds=(lb, ub))
            x = np.array(x)

            if verbosity:
                mse_loss = state.fun_val-.5*LBFGS_rho_x0*np.sum(x**2)
                print(
                    f"\nFinal loss MSE (after LBFGS refinement) = {mse_loss: 8.6f}")
        return x

    def simulate(self, xz0, U):
        """Simulates the model on a test data. Automatic normalization and back-scaling is applied. For simulation,
        only the process part is evaluated.

        Args:
            x0 (ndarray or list of ndarrays) : Initial state to start the simulates, must be (model.nx,) shaped.
                If the model is evaluated on multiple data sequences at the same time, this must be a list, containing
                the estimated initial states.
            U (ndarray ir list of ndarrays) : Input for simulation as an N-by-nu numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-nu numpy arrays.

        Returns:
            Y_back_scaled (ndarray or list of ndarrays) : Back-scaled simulated output for each data sequence.
            X (ndarray or list of ndarrays) : Simulated states for each data sequence.
        """
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])

        @jax.jit
        def model_step(x, u):
            x_plus = self.state_fcn(x, u, self.params).reshape(-1)
            yhat = self.output_fcn(x, u, self.params)
            return x_plus, yhat

        if isinstance(U_norm, list):
            N_meas = len(U_norm)
            Y = []
            X = []
            for i in range(N_meas):
                xz0_i = xz0[i].copy().reshape(-1)
                x = xz0_i[:self.nx_x]
                u = vec_reshape(U_norm[i])
                _, YX = jax.lax.scan(model_step, x, u)
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        else:
            xz = xz0.copy().reshape(-1)
            x = xz0[:self.nx_x]
            _, YX = jax.lax.scan(model_step, x, vec_reshape(U_norm))
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]

        Y_back_scaled = back_scale_data(Y, self.norm['y_mean'], self.norm['y_std'])
        return Y_back_scaled, X

    def predict_one_step_ahead(self, xz0: np.ndarray | jnp.ndarray | list, U: np.ndarray | list, Y: np.ndarray | list):
        """Predicts the model output on a test data in a one-ste-ahead manner. Automatic normalization and back-scaling
        is applied. The current output values are only utilized for predicting the next state values with the innovation
        noise structure.

        Args:
            xz0 (ndarray or list of ndarrays) : Combined initial process and noise state to start the predictions,
                must be (model.nx,) shaped. If the model is evaluated on multiple data sequences at the same time, this
                must be a list, containing the estimated initial states.
            U (ndarray ir list of ndarrays) : Input for predictions as an N-by-nu numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-nu numpy arrays.
            Y (ndarray or list of ndarrays) : Measured output values as an N-by-ny numpy array. If the model is evaluated
                on multiple data sequences at the same time, this must be a list of Ni-by-ny numpy arrays.

        Returns:
            Y_back_scaled (ndarray or list of ndarrays) : Back-scaled simulated output for each data sequence.
            X (ndarray or list of ndarrays) : Simulated states for each data sequence.
        """
        U_norm, _, _ = normalize_data(U, self.norm['u_mean'], self.norm['u_std'])
        Y_norm, _, _ = normalize_data(Y, self.norm['y_mean'], self.norm['y_std'])

        @jax.jit
        def model_step(xz, uy):
            x = xz[:self.nx_x]
            z = xz[self.nx_x:]
            u = uy[:self.nu]
            y = uy[self.nu:]
            x_plus = self.state_fcn(x, u, self.params)
            yhat = self.output_fcn(x, u, self.params)
            z_plus = self.noise_state_fcn(z, x, u, y - yhat, self.params)
            yhat = yhat - self.noise_output_fcn(z, x, u, self.params)
            xz = jnp.hstack((x_plus, z_plus))
            return xz, yhat

        if isinstance(U_norm, list):
            N_meas = len(U_norm)
            Y = []
            X = []
            for i in range(N_meas):
                x = xz0[i].copy().reshape(-1)
                u = vec_reshape(U_norm[i])
                y = vec_reshape(Y_norm[i])
                uy = np.hstack((u, y))
                _, YX = jax.lax.scan(model_step, x, uy)
                Y.append(YX[:, 0:self.ny])
                X.append(YX[:, self.ny:])
        else:
            x = xz0.copy().reshape(-1)
            UY = np.hstack((vec_reshape(U_norm), vec_reshape(Y_norm)))
            _, YX = jax.lax.scan(model_step, x, UY)
            Y = YX[:, 0:self.ny]
            X = YX[:, self.ny:]

        Y_back_scaled = back_scale_data(Y, self.norm['y_mean'], self.norm['y_std'])
        return Y_back_scaled, X
