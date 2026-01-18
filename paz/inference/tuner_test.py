import jax
import jax.numpy as jp

from paz.inference.tuner import AcceptanceToVariance, Tuner


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
    _, infos = tune(key, num_steps, num_episodes, 0.1)
    assert infos.acceptance_rate.shape[0] == num_episodes


def test_tuner_sigma_is_finite():
    key = jax.random.PRNGKey(1)
    num_chains = 2
    positions = jp.zeros((num_chains,))

    def log_density_fn(position):
        return -jp.sum(position ** 2)

    tune = Tuner(log_density_fn, positions, num_chains, progress=False)
    tuned_sigma, _ = tune(key, 1, 1, 0.1)
    assert jp.isfinite(tuned_sigma)
