import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

# Assuming the BayesianLinearRegression class from the previous answer is defined here
# (I've included it again for this code block to be self-contained)

import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.scipy.linalg


class BayesianLinearRegression:
    def __init__(self, initial_alpha=1.0, initial_N0_factor=1.0):
        self.alpha = initial_alpha
        self.N0_factor = initial_N0_factor
        self.A_mean = None
        self.V_hat = None
        self.K_A_post_cols_inv_prior_related = None
        self.S_y_x_plus_S0 = None
        self.df_predictive = None
        self.d = None
        self.m = None
        self.N = None
        self.fitted_N0 = None
        self.fitted_alpha = None

    def _calculate_S_y_x(self, Y, X, XXT, alpha_current):
        # Implements S_y|x = YY^T - (alpha+1)^{-1} YX^T (XX^T)^{-1} XY^T (eq. 33)
        XYT = X @ Y.T
        try:
            Q = jax.scipy.linalg.solve(XXT, XYT, assume_a="pos")
        except jnp.linalg.LinAlgError:
            reg_val = 1e-6 * jnp.mean(jnp.diag(XXT))
            if reg_val < 1e-9:
                reg_val = 1e-6
            reg = jnp.eye(self.m) * reg_val
            XXT_reg = XXT + reg
            Q = jax.scipy.linalg.solve(XXT_reg, XYT, assume_a="pos")

        term_to_subtract = Y @ X.T @ Q

        S_y_x = Y @ Y.T - (1.0 / (alpha_current + 1.0)) * term_to_subtract
        return S_y_x

    def fit(
        self,
        X_train,
        Y_train,
        iterations=10,
        V_known=None,
        min_alpha=1e-6,
        min_N0_add=0.01,
    ):
        self.d = Y_train.shape[0]
        self.m = X_train.shape[0]
        self.N = Y_train.shape[1]

        if self.m > self.N:
            pass

        XXT = X_train @ X_train.T

        current_alpha = self.alpha
        current_N0 = float(self.d - 1 + self.N0_factor)
        if current_N0 <= self.d - 1:
            current_N0 = float(self.d - 1 + min_N0_add)

        if V_known is not None:
            self.V_hat = V_known
            try:
                V_inv = jnp.linalg.inv(self.V_hat)
            except jnp.linalg.LinAlgError:
                jitter_v = (
                    jnp.eye(self.d) * 1e-6 * jnp.mean(jnp.diag(self.V_hat))
                )
                V_inv = jnp.linalg.inv(self.V_hat + jitter_v)

            XYT = X_train @ Y_train.T
            try:
                Q_alpha_opt = jax.scipy.linalg.solve(XXT, XYT, assume_a="pos")
            except jnp.linalg.LinAlgError:
                reg_val = 1e-6 * jnp.mean(jnp.diag(XXT))
                if reg_val < 1e-9:
                    reg_val = 1e-6
                reg = jnp.eye(self.m) * reg_val
                Q_alpha_opt = jax.scipy.linalg.solve(
                    XXT + reg, XYT, assume_a="pos"
                )

            trace_term = jnp.trace(V_inv @ Y_train @ X_train.T @ Q_alpha_opt)

            alpha_numerator = self.m * self.d
            alpha_denominator = trace_term - self.m * self.d

            if alpha_denominator > 1e-9:
                current_alpha = alpha_numerator / alpha_denominator

            current_alpha = jnp.maximum(current_alpha, min_alpha)
            self.fitted_alpha = current_alpha
            self.fitted_N0 = None
        else:
            for i in range(iterations):
                S_y_x_current = self._calculate_S_y_x(
                    Y_train, X_train, XXT, current_alpha
                )
                V_hat_current = (
                    S_y_x_current + current_N0 * jnp.eye(self.d)
                ) / (self.N + current_N0)
                V_hat_current = (V_hat_current + V_hat_current.T) / 2.0

                diag_mean_V = jnp.mean(jnp.diag(V_hat_current))
                if diag_mean_V < 1e-9:
                    diag_mean_V = 1.0
                jitter_v_iter = jnp.eye(self.d) * 1e-7 * diag_mean_V

                try:
                    V_hat_inv = jnp.linalg.inv(V_hat_current + jitter_v_iter)
                except jnp.linalg.LinAlgError:
                    V_hat_inv = jnp.eye(self.d)

                XYT_alpha = X_train @ Y_train.T
                try:
                    Q_alpha = jax.scipy.linalg.solve(
                        XXT, XYT_alpha, assume_a="pos"
                    )
                except jnp.linalg.LinAlgError:
                    reg_val = 1e-6 * jnp.mean(jnp.diag(XXT))
                    if reg_val < 1e-9:
                        reg_val = 1e-6
                    reg = jnp.eye(self.m) * reg_val
                    Q_alpha = jax.scipy.linalg.solve(
                        XXT + reg, XYT_alpha, assume_a="pos"
                    )

                trace_term_alpha = jnp.trace(
                    V_hat_inv @ Y_train @ X_train.T @ Q_alpha
                )
                alpha_numerator = self.m * self.d
                alpha_denominator = trace_term_alpha - self.m * self.d

                if alpha_denominator > 1e-9:
                    new_alpha = alpha_numerator / alpha_denominator
                    current_alpha = jnp.maximum(new_alpha, min_alpha)

                idx = jnp.arange(1, self.d + 1)
                digamma_term1_args = (self.N + current_N0 + 1.0 - idx) / 2.0
                digamma_term2_args = (current_N0 + 1.0 - idx) / 2.0

                if jnp.any(digamma_term1_args <= 1e-9) or jnp.any(
                    digamma_term2_args <= 1e-9
                ):
                    new_N0 = current_N0
                else:
                    sum_digamma_diff = jnp.sum(
                        jax.scipy.special.digamma(digamma_term1_args)
                        - jax.scipy.special.digamma(digamma_term2_args)
                    )
                    mat_for_det = S_y_x_current / current_N0 + jnp.eye(self.d)
                    sign, log_det_val = jnp.linalg.slogdet(mat_for_det)

                    if sign <= 1e-9:
                        new_N0 = current_N0
                    else:
                        log_det_term = log_det_val
                        trace_V_inv_term = jnp.trace(V_hat_inv)
                        N0_update_numerator = (
                            current_N0 + 1.0 - self.d
                        ) * sum_digamma_diff
                        N0_update_denominator = (
                            log_det_term + trace_V_inv_term - self.d
                        )

                        if jnp.abs(N0_update_denominator) < 1e-9:
                            new_N0 = current_N0
                        else:
                            new_N0 = (
                                self.d - 1.0
                            ) + N0_update_numerator / N0_update_denominator

                current_N0 = jnp.maximum(new_N0, float(self.d - 1 + min_N0_add))
                self.V_hat = V_hat_current

            self.fitted_alpha = current_alpha
            self.fitted_N0 = current_N0
            S_y_x_final = self._calculate_S_y_x(
                Y_train, X_train, XXT, self.fitted_alpha
            )
            self.V_hat = (S_y_x_final + self.fitted_N0 * jnp.eye(self.d)) / (
                self.N + self.fitted_N0
            )
            self.V_hat = (self.V_hat + self.V_hat.T) / 2.0

        self.K_A_post_cols_inv_prior_related = XXT * (self.fitted_alpha + 1.0)

        try:
            diag_mean_K = jnp.mean(
                jnp.diag(self.K_A_post_cols_inv_prior_related)
            )
            if diag_mean_K < 1e-9:
                diag_mean_K = 1.0
            jitter_k = jnp.eye(self.m) * 1e-7 * diag_mean_K
            self.A_mean = (
                Y_train
                @ X_train.T
                @ jnp.linalg.inv(
                    self.K_A_post_cols_inv_prior_related + jitter_k
                )
            )
        except jnp.linalg.LinAlgError:
            self.A_mean = (
                Y_train
                @ X_train.T
                @ jnp.linalg.pinv(self.K_A_post_cols_inv_prior_related)
            )

        if V_known is None:
            S_y_x_for_pred = self._calculate_S_y_x(
                Y_train, X_train, XXT, self.fitted_alpha
            )
            S0_final = self.fitted_N0 * jnp.eye(self.d)
            self.S_y_x_plus_S0 = S_y_x_for_pred + S0_final
            self.df_predictive = self.N + self.fitted_N0 + 1.0
        else:
            self.S_y_x_plus_S0 = None
            self.df_predictive = None

    def predict(self, X_test):
        if self.A_mean is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        N_test = X_test.shape[1]
        Y_pred_mean = self.A_mean @ X_test

        Y_pred_cov_params = []

        for i in range(N_test):
            x_t = X_test[:, i : i + 1]
            try:
                diag_mean_K_pred = jnp.mean(
                    jnp.diag(self.K_A_post_cols_inv_prior_related)
                )
                if diag_mean_K_pred < 1e-9:
                    diag_mean_K_pred = 1.0
                jitter_k_pred = jnp.eye(self.m) * 1e-7 * diag_mean_K_pred
                S_xx_inv_xt = jax.scipy.linalg.solve(
                    self.K_A_post_cols_inv_prior_related + jitter_k_pred,
                    x_t,
                    assume_a="pos",
                )
            except jnp.linalg.LinAlgError:
                S_xx_inv_xt = (
                    jnp.linalg.pinv(self.K_A_post_cols_inv_prior_related) @ x_t
                )

            c_inv_scalar = (1.0 + x_t.T @ S_xx_inv_xt)[0, 0]

            if (
                self.S_y_x_plus_S0 is not None
                and self.df_predictive is not None
            ):
                scale_matrix = self.S_y_x_plus_S0 * c_inv_scalar
                df = self.df_predictive
                Y_pred_cov_params.append(
                    {
                        "type": "student_t",
                        "scale_matrix": scale_matrix,
                        "df": df,
                    }
                )
            else:
                covariance_matrix = self.V_hat * c_inv_scalar
                Y_pred_cov_params.append(
                    {"type": "gaussian", "covariance": covariance_matrix}
                )

        return Y_pred_mean, Y_pred_cov_params


# 1. Generate Synthetic Data
# Use a JAX PRNG key for reproducibility
key = random.PRNGKey(0)
key, subkey = random.split(key)

# Define dimensions and number of samples
d = 1  # 1D output
m = 2  # 1 feature + 1 intercept
N = 30  # Number of training samples
N_test = 100  # Number of test points for plotting

# Define the true parameters
A_true = jnp.array([[2.5, -1.2]])  # True intercept and slope
V_true_std = 2.0  # Standard deviation of the noise
V_true = jnp.array([[V_true_std**2]])

# Generate training data
x_train_coords = random.uniform(subkey, shape=(N,), minval=-5, maxval=5)
key, subkey = random.split(key)
# Add a row of ones for the intercept term
X_train = jnp.vstack([jnp.ones(N), x_train_coords])
# Generate noise
noise = random.normal(subkey, shape=(d, N)) * V_true_std
# Generate Y_train using the linear model equation Y = AX + noise
Y_train = A_true @ X_train + noise

# Generate test data (a grid of points for plotting)
x_test_coords = jnp.linspace(-8, 8, N_test)
X_test = jnp.vstack([jnp.ones(N_test), x_test_coords])


# 2. Fit the Model (assuming V is unknown)
model = BayesianLinearRegression()
model.fit(X_train, Y_train, iterations=20)


# 3. Predict on the test data
Y_pred_mean, Y_pred_cov_params = model.predict(X_test)


# 4. Visualize the results
# Convert JAX arrays to NumPy arrays for plotting
np_X_train = np.asarray(X_train)
np_Y_train = np.asarray(Y_train)
np_X_test_coords = np.asarray(x_test_coords)
np_Y_pred_mean = np.asarray(Y_pred_mean)

# Calculate the standard deviation for the credible interval
pred_std = []
for params in Y_pred_cov_params:
    if params["type"] == "student_t":
        # For Student-T, variance = scale * df / (df - 2)
        scale = np.asarray(params["scale_matrix"])[0, 0]  # d=1
        df = params["df"]
        # The variance is only defined for df > 2
        if df > 2:
            var = scale * df / (df - 2.0)
            pred_std.append(np.sqrt(var))
        else:
            # If df <= 2, variance is infinite. We can use a large number
            # or just show the mean. Let's use the scale parameter as a proxy.
            pred_std.append(np.sqrt(scale))

    elif params["type"] == "gaussian":
        # For Gaussian, variance is the diagonal of the covariance matrix
        cov = np.asarray(params["covariance"])[0, 0]  # d=1
        pred_std.append(np.sqrt(cov))

np_pred_std = np.array(pred_std)

# Get the y-values for the true line
Y_true_line = np.asarray(A_true @ X_test)


# Create the plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the training data
ax.scatter(
    np_X_train[1, :],
    np_Y_train[0, :],
    c="blue",
    label="Training Data",
    zorder=3,
)

# Plot the true line
ax.plot(np_X_test_coords, Y_true_line[0, :], "k--", label="True Line", zorder=2)

# Plot the predicted mean
ax.plot(
    np_X_test_coords,
    np_Y_pred_mean[0, :],
    "r-",
    label="Predicted Mean",
    zorder=2,
)

# Plot the 95% credible interval (mean +/- 1.96 * std)
ax.fill_between(
    np_X_test_coords,
    np_Y_pred_mean[0, :] - 1.96 * np_pred_std,
    np_Y_pred_mean[0, :] + 1.96 * np_pred_std,
    color="red",
    alpha=0.2,
    label="95% Credible Interval",
)

# Final plot settings
ax.set_title("Bayesian Linear Regression (V Unknown)", fontsize=16)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
