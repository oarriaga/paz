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
        # Implements S_y|x = YY^T - (alpha+1)^{-1} YX^T (XX^T)^{-1} XY^T (eq. 33) [cite: 37]
        # Y (d,N), X (m,N), XXT (m,m)

        # Term YX^T (XX^T)^{-1} XY^T
        # Safely compute (XX^T)^{-1} XY^T as solve(XXT, XY^T)
        # XY_T is Y @ X.T, but here it's X @ Y.T if we need (XX^T)^-1 (X Y^T)
        # The term is Y X^T (X X^T)^{-1} X Y^T. Let Z = X Y^T. Solve (X X^T) Q = Z for Q. Then Y X^T Q.
        # Or, (Y X^T) @ solve(X X^T, X Y^T).

        XYT = X @ Y.T  # m x d
        try:
            # Solve XXT @ Q = XYT for Q
            Q = jax.scipy.linalg.solve(XXT, XYT, assume_a="pos")
        except jnp.linalg.LinAlgError:
            # Fallback for singular XXT: add regularization
            reg_val = 1e-6 * jnp.mean(jnp.diag(XXT))
            if reg_val < 1e-9:  # If diag mean is zero (e.g. XXT is zero matrix)
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
            # This situation means XXT is singular.
            # The _calculate_S_y_x and other parts with solve(XXT,...) will attempt to handle it.
            # The paper's invariant prior (eq. 16) implies alpha*XXT is non-singular[cite: 25],
            # which means XXT itself should be non-singular if alpha > 0.
            # Eq. 33 also uses (XXT)^-1[cite: 37].
            pass  # Handled by regularized solves

        XXT = X_train @ X_train.T  # m x m

        current_alpha = self.alpha
        # Initialize N0 > d-1 for proper Inverse Wishart prior
        # N0 must be > d-1 for Gamma functions in Z_nd to be defined (eq. 47, 49) [cite: 65]
        current_N0 = float(self.d - 1 + self.N0_factor)
        if current_N0 <= self.d - 1:  # Ensure strictly > d-1
            current_N0 = float(self.d - 1 + min_N0_add)

        if V_known is not None:
            self.V_hat = V_known
            # Optimize alpha using eq. 35 if V is known [cite: 42]
            # alpha = (m*d) / (tr(V_inv @ YX^T @ (XX^T)^-1 @ XY^T) - m*d)
            try:
                V_inv = jnp.linalg.inv(self.V_hat)
            except jnp.linalg.LinAlgError:
                jitter_v = (
                    jnp.eye(self.d) * 1e-6 * jnp.mean(jnp.diag(self.V_hat))
                )
                V_inv = jnp.linalg.inv(self.V_hat + jitter_v)

            XYT = X_train @ Y_train.T  # m x d
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

            if (
                alpha_denominator <= 1e-9
            ):  # Avoid division by zero or negative alpha
                # Keep initial or last alpha if denominator is not positive
                pass  # current_alpha remains self.alpha
            else:
                current_alpha = alpha_numerator / alpha_denominator

            current_alpha = jnp.maximum(current_alpha, min_alpha)
            self.fitted_alpha = current_alpha
            self.fitted_N0 = None
        else:  # V is unknown, iterate to find alpha, N0, V_hat [cite: 67]
            for i in range(iterations):
                # 1. Calculate S_y|x (eq. 33) using current_alpha
                S_y_x_current = self._calculate_S_y_x(
                    Y_train, X_train, XXT, current_alpha
                )

                # 2. Estimate V_hat (eq. 57)
                V_hat_current = (
                    S_y_x_current + current_N0 * jnp.eye(self.d)
                ) / (self.N + current_N0)
                V_hat_current = (
                    V_hat_current + V_hat_current.T
                ) / 2.0  # Ensure symmetry

                # Add jitter to V_hat_current before inversion if it's ill-conditioned
                diag_mean_V = jnp.mean(jnp.diag(V_hat_current))
                if diag_mean_V < 1e-9:
                    diag_mean_V = 1.0  # handle zero matrix V_hat
                jitter_v_iter = jnp.eye(self.d) * 1e-7 * diag_mean_V

                # 3. Update alpha (eq. 58)
                try:
                    V_hat_inv = jnp.linalg.inv(V_hat_current + jitter_v_iter)
                except jnp.linalg.LinAlgError:  # Should be rare due to jitter
                    V_hat_inv = jnp.eye(self.d)  # Fallback

                XYT_alpha = X_train @ Y_train.T  # m x d
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

                if alpha_denominator <= 1e-9:
                    new_alpha = current_alpha
                else:
                    new_alpha = alpha_numerator / alpha_denominator
                current_alpha = jnp.maximum(new_alpha, min_alpha)

                # 4. Update N0 (eq. 59)
                idx = jnp.arange(1, self.d + 1)
                digamma_term1_args = (self.N + current_N0 + 1.0 - idx) / 2.0
                digamma_term2_args = (current_N0 + 1.0 - idx) / 2.0

                # Ensure arguments for digamma are positive
                if jnp.any(digamma_term1_args <= 1e-9) or jnp.any(
                    digamma_term2_args <= 1e-9
                ):
                    new_N0 = current_N0  # Skip update if args are bad
                else:
                    sum_digamma_diff = jnp.sum(
                        jax.scipy.special.digamma(digamma_term1_args)
                        - jax.scipy.special.digamma(digamma_term2_args)
                    )

                    mat_for_det = S_y_x_current / current_N0 + jnp.eye(self.d)
                    sign, log_det_val = jnp.linalg.slogdet(mat_for_det)

                    if sign <= 1e-9:  # Determinant is zero or negative
                        new_N0 = current_N0  # Skip update
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
                            new_N0 = (
                                current_N0  # Skip update if denominator is zero
                            )
                        else:
                            new_N0 = (
                                self.d - 1.0
                            ) + N0_update_numerator / N0_update_denominator

                current_N0 = jnp.maximum(new_N0, float(self.d - 1 + min_N0_add))
                self.V_hat = V_hat_current

            self.fitted_alpha = current_alpha
            self.fitted_N0 = current_N0
            # Final V_hat with optimized alpha and N0
            S_y_x_final = self._calculate_S_y_x(
                Y_train, X_train, XXT, self.fitted_alpha
            )
            self.V_hat = (S_y_x_final + self.fitted_N0 * jnp.eye(self.d)) / (
                self.N + self.fitted_N0
            )
            self.V_hat = (self.V_hat + self.V_hat.T) / 2.0

        # --- Store parameters for prediction ---
        # S_xx for posterior of A under invariant prior (eq. 27) is XX^T(alpha+1)
        self.K_A_post_cols_inv_prior_related = XXT * (self.fitted_alpha + 1.0)

        # A_mean = YX^T @ S_xx_inv = YX^T @ (XX^T(alpha+1))^{-1} (from eq. 27) [cite: 27]
        try:
            # Add jitter to K_A_post_cols_inv_prior_related before inversion
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
        except jnp.linalg.LinAlgError:  # Should be rare with jitter
            self.A_mean = (
                Y_train
                @ X_train.T
                @ jnp.linalg.pinv(self.K_A_post_cols_inv_prior_related)
            )

        if V_known is None:  # V was unknown
            S_y_x_for_pred = self._calculate_S_y_x(
                Y_train, X_train, XXT, self.fitted_alpha
            )
            S0_final = self.fitted_N0 * jnp.eye(self.d)
            self.S_y_x_plus_S0 = S_y_x_for_pred + S0_final
            # Degrees of freedom for predictive T (eq. 60) [cite: 67]
            self.df_predictive = self.N + self.fitted_N0 + 1.0
        else:
            self.S_y_x_plus_S0 = None
            self.df_predictive = None

    def predict(self, X_test):
        if self.A_mean is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        N_test = X_test.shape[1]
        # Predictive mean E[y|x,D] = A_mean @ x_test (from eq. 57 where E[A|D] is used) [cite: 57]
        Y_pred_mean = self.A_mean @ X_test

        Y_pred_cov_params = []

        for i in range(N_test):
            x_t = X_test[:, i : i + 1]  # m x 1

            # c = (1 + x_t^T @ S_xx_inv @ x_t)^{-1} (eq. 41) [cite: 57]
            # S_xx for invariant prior is self.K_A_post_cols_inv_prior_related
            try:
                # Add jitter for robustness of solve
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
            except jnp.linalg.LinAlgError:  # Should be rare
                S_xx_inv_xt = (
                    jnp.linalg.pinv(self.K_A_post_cols_inv_prior_related) @ x_t
                )

            # c_inv_scalar is (1/c) from the paper's definition of c
            c_inv_scalar = (1.0 + x_t.T @ S_xx_inv_xt)[0, 0]

            if (
                self.S_y_x_plus_S0 is not None
                and self.df_predictive is not None
            ):  # V was unknown
                # Predictive is Student-T p(y|x,D) (eq. 60) [cite: 67]
                # Scale matrix parameter for T: (S_y|x + S0) * c_inv_scalar
                scale_matrix = self.S_y_x_plus_S0 * c_inv_scalar
                df = self.df_predictive
                Y_pred_cov_params.append(
                    {
                        "type": "student_t",
                        "scale_matrix": scale_matrix,
                        "df": df,
                    }
                )
            else:  # V was known
                # Predictive is Gaussian p(y|x,D,V) (eq. 40) [cite: 57, 58]
                # Covariance: V_hat * c_inv_scalar
                covariance_matrix = self.V_hat * c_inv_scalar
                Y_pred_cov_params.append(
                    {"type": "gaussian", "covariance": covariance_matrix}
                )

        return Y_pred_mean, Y_pred_cov_params


model = BayesianLinearRegression(initial_alpha=1.0, initial_N0_factor=1.0)
# X_train should be an m x N JAX array (m features, N samples)
# Y_train should be a d x N JAX array (d output dimensions, N samples)
# V_known can be a d x d JAX array if noise covariance is known, otherwise None
model.fit(X_train, Y_train, iterations=20, V_known=None)

# X_test should be an m x N_test JAX array
Y_pred_mean, Y_pred_cov_params = model.predict(X_test)
