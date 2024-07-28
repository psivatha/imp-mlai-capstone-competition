import math
import warnings
from dataclasses import dataclass

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement, qKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.quasirandom import SobolEngine

# Suppress the specific warning about input data standardization
warnings.filterwarnings("ignore")

@dataclass
class TurboState:
    """
    The state of the optimisation process.
    Includes parameters like the current best value, the length scale, success and failure counters
    """
    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    """
    Updates th state of the optimisation state with the observed values.
    """
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(state, model, X, Y, batch_size=1, n_candidates=None, num_restarts=10, raw_samples=512, acqf="ei"):
    """
    Generates the next batch to be sent for evaluation
    The default acquisition function is ei and can handle TS, EI and KG
    """
    assert acqf in ["ts", "ei", "kg"]
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim_ = X.shape[-1]
    sobol_ = SobolEngine(dim_, scramble=True)
    pert = sobol_.draw(n_candidates)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim_, 1.0)
    mask = (torch.rand(n_candidates, dim_) <= prob_perturb)
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim_ - 1, size=(len(ind),))] = 1

    X_cand = x_center.expand(n_candidates, dim_).clone()
    X_cand[mask] = pert[mask]

    model.eval()

    if acqf == "ts":
        posterior_distribution = model(X_cand)
        with torch.no_grad():
            posterior_sample = posterior_distribution.sample()
            X_next_idx = torch.argmax(posterior_sample)
            X_next = X_cand[X_next_idx]
    else:
        acq_function = qExpectedImprovement(model, Y.max()) if acqf == "ei" else qKnowledgeGradient(model, num_fantasies=64)
        X_next, acq_value = optimize_acqf(
            acq_function,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        self.training_iter = 200

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Loading data for other functions
Xs = []
Ys = []

# Load the first set of data
for i in range(1, 9):
    Xs.append(np.load(f"initial_data/function_{i}/initial_inputs.npy"))
    Y_first = np.load(f"initial_data/function_{i}/initial_outputs.npy")
    if i == 4:
        Y_first = np.abs(Y_first)
    Ys.append(Y_first)

# Load the first set of data
for i in range(1, 9):
    X_new = np.load(f"initial_data2/function_{i}/initial_inputs.npy")
    Y_new = np.load(f"initial_data2/function_{i}/initial_outputs.npy")
    if i == 4:
        Y_new = np.abs(Y_new)
    Xs[i - 1] = np.vstack((Xs[i - 1], X_new))
    Ys[i - 1] = np.hstack((Ys[i - 1], Y_new))

# Load the feedback data for further training
df = pd.read_csv('feedback_data/605_data.csv')
df = df.drop(columns=['Unnamed: 0', 'timestamp', 'student_id'], axis=1)

for index, col in enumerate(df.columns):
    if 'output' in col:
        y_array = np.array(df[col], dtype=float)
        Ys[index-8] = np.hstack((y_array, Ys[index-8]))
        continue
    series = []
    for i in range(len(df[col])):
        series.append(df[col][i].replace("[", "").replace("]", "").split())
    series_array = np.array(series, dtype=float)
    Xs[index] = np.vstack((series_array, Xs[index]))


# Function 3
# Standardize data for function_3 - drug discovery
scaler = StandardScaler()
X_function_3 = Xs[2]
X_function_3_standardized = scaler.fit_transform(X_function_3)

# Feature selection using Lasso (L1 regularization)
lasso = Lasso(alpha=0.01)
lasso.fit(X_function_3_standardized, Ys[2])
model_1 = SelectFromModel(lasso, prefit=True)
X_function_3_selected = model_1.transform(X_function_3_standardized)

min_max_scaler = MinMaxScaler()
X_function_3_selected_normalized = min_max_scaler.fit_transform(X_function_3_selected)

# Update Xs with the selected and normalized features for function_3
Xs[2] = X_function_3_selected_normalized

print(f"Function 3 feature selection coeffs {lasso.coef_}")

# Extract the last row from the dataframe for updating the state
last_row = df.iloc[-1]

# Default
acquisition_fn = 'ei'

dimensions = [2, 2, 3, 4, 4, 5, 6, 8]
for i, dim in enumerate(dimensions):
    # 7th July - Different acquisition functions are selected depending on the function types.
    if i in [0, 2, 3, 7]:
        # Thomson Sampling
        # Function 3: Drug discovery: Avoid local optima, complexity & noise due to the interaction between components
        # Function 4: Avoid local optima, Exploration focused acquisition fn
        # Function 8: High dimensionality, Local exploration
        acquisition_fn = 'ts'
    elif i in [1, 4, 6]:
        # Expected Improvement
        # Function 5: Focus more on exploitation, Faster convergence in unimodal setup, Some degree of exploration
        # Function 7: Generic reasons for better handling noisy observations, and suitable for bb optimisations
        acquisition_fn = 'ei'
    elif i in [5]:
        # Knowledge gradient
        # Function 6: For better accuracy, balance of Exploration and Exploitation and Multi-objective optimisation
        acquisition_fn = 'kg'

    # Obtain the last received evaluated function values.
    if i < 8:
        last_x = np.array([float(x) for x in last_row[f'f{i+1}'].replace("[", "").replace("]", "").split()])
        last_y = float(last_row[f'f{i+1}_output'])
    else:
        last_x = np.array([float(x) for x in last_row[f'f{i-7}'].replace("[", "").replace("]", "").split()])
        last_y = float(last_row[f'f{i-7}_output'])

    X = torch.tensor(Xs[i], dtype=torch.float32).requires_grad_()
    Y = torch.tensor(Ys[i], dtype=torch.float32).squeeze()

    state = TurboState(dim=dim)
    model = None
    likelihood = None

    # Different acquisition function works well with different surrogate model
    if acquisition_fn == "ts":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X, Y, likelihood)
    elif acquisition_fn in ["ei", "kg"]:
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X, Y.unsqueeze(-1), covar_module=covar_module, likelihood=likelihood
        )

    # Train the model with the available data.
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if acquisition_fn == 'ts':
        for _ in range(model.training_iter):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, Y).sum()
            loss.backward()
            optimizer.step()

    # Update the state with the best value found so far
    state.best_value = max(Ys[i])
    state = update_state(state, torch.tensor([last_y], dtype=torch.float32))

    model.eval()
    likelihood.eval()

    if i == 2:
        # We're transforming Function 3's features so that one of the features has no impact on function value
        # ignore the first feature . see line 215
        X = X[:, -2:]

    next_point = generate_batch(state, model, X, Y, acqf=acquisition_fn)
    if i in [0, 2, 3, 7]:
        next_query_string = "-".join([f"{x:.6f}" for x in next_point.detach().numpy().tolist()])
        if i ==2:
            # We're excluding first feature from training
            next_query_string = "0.000000-" + next_query_string
    else:
        next_query_string = "-".join([f"{x:.6f}" for x in next_point[0]])
    print(f"\t\tThe next query for the function {i+1} is {next_query_string}")

