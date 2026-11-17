import jax
import jax.numpy as jp

from paz.inference.tuner import AcceptanceToVariance, Tuner
from paz.inference.tuner import compute_cholesky_shape, compute_covariance
from paz.inference.tuner import compute_diagonal_shape


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


def tune_correlated(compute_shape):
    log_density_fn = build_correlated_log_density(jp.array([1.0, 0.1]), 0.9)
    num_chains = 8
    positions = jp.zeros((num_chains, 2))
    kwargs = {"progress": False, "compute_shape": compute_shape}
    tune = Tuner(log_density_fn, positions, num_chains, **kwargs)
    return tune(jax.random.PRNGKey(0), 500, 10, 0.1)


def test_diagonal_tuner_recovers_scale_ratio():
    sigma, _, _ = tune_correlated(compute_diagonal_shape)
    assert sigma.shape == (2,)
    assert 5.0 < sigma[0] / sigma[1] < 20.0


def test_covariance_tuner_recovers_correlation():
    sigma, _, _ = tune_correlated(compute_cholesky_shape)
    covariance = sigma @ sigma.T
    stdvs = jp.sqrt(jp.diagonal(covariance))
    correlation = covariance[0, 1] / (stdvs[0] * stdvs[1])
    assert sigma.shape == (2, 2)
    assert jp.abs(correlation - 0.9) < 0.1


def test_tuner_returns_last_positions():
    _, positions, _ = tune_correlated(compute_cholesky_shape)
    assert positions.shape == (8, 2)
    assert jp.all(positions != 0.0)


def test_covariance_of_stuck_chains_is_positive_definite():
    covariance = compute_covariance(jp.ones((4, 10, 3)))
    assert jp.all(jp.isfinite(jp.linalg.cholesky(covariance)))
