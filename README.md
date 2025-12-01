# deepSI-jax
JAX implementation of the [deepSI](https://github.com/MaartenSchoukens/deepSI) toolbox for deep-learning-based identification of dynamical systems. For an efficient identification pipeline, the [jax-sysid](https://github.com/bemporad/jax-sysid) toolbox was adapted to incorporate the SUBNET-based elements.

# Installation
For standalone installation, recommended for day-to-day usage, the most convenient way to install the toolbox is:
```bash
pip install git+https://github.com/AIMotionLab-SZTAKI/deepSI-jax@main
```
(Installation directly from PyPI will be available soon...)

<br>

Alternatively, to be able to run the example scripts and/or modify the codebase, clone the repository:
```bash
git clone https://github.com/AIMotionLab-SZTAKI/deepSI-jax
```
Then, install the package and its dependencies (it is advised to use a virtual environment), as
```bash
cd deepSI-jax
pip install -e .
```

# Example usage
```py
import numpy as np
import deepSI_jax
from deepSI_jax.utils import NRMS_error

# Generate or load data
np.random.seed(0)
U = np.random.randn(10_000) # Input sequence
x = [0, 0] # Initial state
ylist = [] # Output sequence
for uk in U:
    ylist.append(x[1]*x[0]*0.1 + x[0] + np.random.randn()*1e-3)  # Compute output
    x = x[0]/(1.2+x[1]**2) + x[1]*0.4, \
        x[1]/(1.2+x[0]**2) + x[0]*0.4 + uk*(1+x[0]**2/10) # Advance state

# Split dataset
Y = np.array(ylist)
Y_train = Y[:9000]
Y_test = Y[9000:]
U_train = U[:9000]
U_test = U[9000:]

# Hyperparameters and data normalization
nu, ny, norm = deepSI_jax.get_nu_ny_and_auto_norm(Y_train, U_train)
nx = 3
n = 20  # state initialization window (encoder lag)

# Create model (with default ANN hyperparameters)
model = deepSI_jax.SUBNET(nx=nx, nu=nu, ny=ny, norm=norm, encoder_lag=n)

# set loss function and optimization parameters
model.set_loss_fun(l2_reg=1e-4, T=200)
model.optimization(adam_epochs=1000, lbfgs_epochs=5000)

# Train model on data
model.fit(Y_train, U_train)

# Simulate model on the test input sequence, but first estimate the initial states with the encoder
x0_test = model.encoder_estim_x0(Y_test[:n], U_test[:n])
Yhat_test, Xhat_test = model.simulate(x0_test, U_test[n:])

nrmse = NRMS_error(Y_test[n:], Yhat_test)
sim_idx = np.arange(U_test.shape[0])

# Visualize simulation of the model
from matplotlib import pyplot as plt
plt.figure(figsize=(7,3))
plt.plot(sim_idx, Y_test, label='Real Data')
plt.plot(sim_idx[n:], Yhat_test, label=f'Model Sim. (NRMS = {nrmse:.2%})', linestyle='--')
plt.title('Comparison of Real Data and Model Simulation', fontsize=14, fontweight='bold')
plt.legend(); plt.xlabel('Time Index'); plt.ylabel('y'); plt.grid(); plt.tight_layout(pad=0.5)
plt.show()
```
<img width="698" height="295" alt="example_usage_figure_for_github" src="https://github.com/user-attachments/assets/736ae06b-a5b8-4303-bcd6-57a76b2a0794" />

# Citing
The `deepSI_jax` toolbox offers an efficient, JAX-based implementation of the subspace encoder method. When using the SUBNET structure, please cite:
```
@article{beintema_deep_2023,
	title = {Deep subspace encoders for nonlinear system identification},
	volume = {156},
	journal = {Automatica},
	author = {Beintema, Gerben I. and Schoukens, Maarten and Tóth, Roland},
	year = {2023},
	pages = {111210}
}
```

The toolbox is built by using the highly efficient `jax-sysid` toolbox, so please also cite:
```
@article{bemporad_l-bfgs-b_2025,
	title = {An {L}-{BFGS}-{B} {Approach} for {Linear} and {Nonlinear} {System} {Identification} {Under} {$\ell_1$} and {Group}-{Lasso} {Regularization}},
	journal = {IEEE Transactions on Automatic Control},
	author = {Bemporad, Alberto},
	year = {2025},
	pages = {4857--4864},
    volume={70},
    number={7}
}
```

# License
See the [LICENSE](/LICENSE) file for license rights and limitations (MIT).
