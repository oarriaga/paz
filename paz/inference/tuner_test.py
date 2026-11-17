import jax
import jax.numpy as jp

from paz.inference.tuner import AcceptanceToVariance, Tuner
from paz.inference.tuner import compute_cholesky_sigma, compute_diagonal_sigma
from paz.inference.tuner import keep_sigma


def test_acceptance_to_variance_returns_finite():
    acceptance = jp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    factors = jp.array([0.1, 0.5, 0.9, 1.1, 2.0, 10.0])
    rate_fn = AcceptanceToVariance(acceptance, factors)
    assert jp.isfinite(rate_fn(0.2))


def test_tuner_returns_infos_shape():
    key = jax.random.PRNGKey(0)
    num_chains = 2
    num_steps = 1
    num_episodes = 2
    positions = jp.zeros((num_chains,))

    def log_density_fn(position):
        return -jp.sum(position ** 2)

    tune = Tuner(log_density_fn, positions, num_chains, progress=False)
    _, _, infos = tune(key, num_steps, num_episodes, 0.1)
    assert infos.acceptance_rate.shape[0] == num_episodes


def test_tuner_sigma_is_finite():
    key = jax.random.PRNGKey(1)
    num_chains = 2
    positions = jp.zeros((num_chains,))

    def log_density_fn(position):
        return -jp.sum(position ** 2)

    tune = Tuner(log_density_fn, positions, num_chains, progress=False)
    tuned_sigma, _, _ = tune(key, 1, 1, 0.1)
    assert jp.isfinite(tuned_sigma)


def build_correlated_log_density(stdvs, correlation):
    correlations = jp.array([[1.0, correlation], [correlation, 1.0]])
    covariance = jp.outer(stdvs, stdvs) * correlations
    precision = jp.linalg.inv(covariance)
    return lambda position: -0.5 * position @ precision @ position


def tune_correlated(compute_sigma):
    log_density_fn = build_correlated_log_density(jp.array([1.0, 0.1]), 0.9)
    num_chains = 8
    positions = jp.zeros((num_chains, 2))
    kwargs = {"progress": False, "compute_sigma": compute_sigma}
    tune = Tuner(log_density_fn, positions, num_chains, **kwargs)
    return tune(jax.random.PRNGKey(0), 500, 10, 0.1)


def test_diagonal_tuner_recovers_scale_ratio():
    sigma, _, _ = tune_correlated(compute_diagonal_sigma)
    assert sigma.shape == (2,)
    assert 5.0 < sigma[0] / sigma[1] < 20.0


def test_covariance_tuner_recovers_correlation():
    sigma, _, _ = tune_correlated(compute_cholesky_sigma)
    covariance = sigma @ sigma.T
    stdvs = jp.sqrt(jp.diagonal(covariance))
    correlation = covariance[0, 1] / (stdvs[0] * stdvs[1])
    assert sigma.shape == (2, 2)
    assert jp.abs(correlation - 0.9) < 0.1


def test_tuner_returns_last_positions():
    _, positions, _ = tune_correlated(compute_cholesky_sigma)
    assert positions.shape == (8, 2)
    assert jp.all(positions != 0.0)


def test_covariance_tuner_keeps_shape_when_chains_never_move():
    def log_density_fn(position):
        return jp.where(jp.all(position == 0.0), 0.0, -jp.inf)

    positions = jp.zeros((4, 2))
    kwargs = {"progress": False, "compute_sigma": compute_cholesky_sigma}
    tune = Tuner(log_density_fn, positions, 4, **kwargs)
    sigma, _, _ = tune(jax.random.PRNGKey(0), 20, 3, 0.1)
    assert jp.all(jp.isfinite(sigma))
    assert jp.all(jp.diagonal(sigma) > 0.0)


def tune_one_episode(compute_sigma, guess):
    def log_density_fn(position):
        return -0.5 * jp.sum(position**2)

    kwargs = {"progress": False, "compute_sigma": compute_sigma}
    tune = Tuner(log_density_fn, jp.zeros((4, 2)), 4, **kwargs)
    sigma, _, _ = tune(jax.random.PRNGKey(0), 20, 1, guess)
    return sigma


def test_step_tuner_keeps_proportions_of_vector_guess():
    guess = jp.array([0.5, 0.05])
    sigma = tune_one_episode(keep_sigma, guess)
    assert jp.allclose(sigma / sigma[0], guess / guess[0])


def test_diagonal_tuner_starts_from_vector_guess():
    guess = jp.array([0.5, 0.05])
    sigma = tune_one_episode(compute_diagonal_sigma, guess)
    assert jp.allclose(sigma / sigma[0], guess / guess[0], rtol=1e-4)


def test_covariance_tuner_starts_from_matrix_guess():
    guess = jp.array([[0.5, 0.0], [0.2, 0.05]])
    sigma = tune_one_episode(compute_cholesky_sigma, guess)
    assert jp.allclose(sigma / sigma[0, 0], guess / guess[0, 0], atol=1e-4)
