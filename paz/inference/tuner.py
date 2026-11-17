from collections import namedtuple
from functools import partial

import jax
from jax import flatten_util
import jax.numpy as jp

import paz.utils.progressbar as progressbar
from .metropolis_hastings import (
    Samples,
    build_new_proposal,
    build_now_proposal,
    choose_proposal,
    propose_additively,
)


TunerInfo = namedtuple("TunerInfo", ["scale", "factor", "acceptance_rate"])
TunerState = namedtuple("TunerState", ["kernel_state", "scale", "factor"])
EpisodeState = namedtuple("EpisodeState", ["key", "tuner_state", "sigma"])
Episode = namedtuple("Episode", ["infos", "sigma", "is_sigma_valid"])
Chain = namedtuple("Chain", ["info", "positions"])
result_fields = ["sigma", "positions", "infos", "is_sigma_valid"]
TunerResult = namedtuple("TunerResult", result_fields)
fields = ["sigma", "num_steps", "num_episodes", "progress", "compute_rate"]
TunerConfig = namedtuple("TunerConfig", fields + ["compute_sigma"])
ACCEPTANCE_RATES = jp.array([0.001, 0.05, 0.2, 0.5, 0.75, 0.95])
VARIANCE_FACTORS = jp.array([0.1, 0.5, 0.9, 1.1, 2, 10])


def AdaptiveStepTuner(
    sigma, num_steps=200, num_episodes=10, progress=True, compute_rate=None
):
    """Tunes the size of the proposal step for Metropolis-Hastings.

    The proposal is `scale * sigma`. The scale follows the acceptance rate.
    `sigma` keeps the proportions it was given.

    # Arguments
        sigma: Initial proposal sigma. A number, one value per parameter,
            or a Cholesky factor of the proposal covariance.
        num_steps: Int. Steps per chain in each episode.
        num_episodes: Int. Episodes; the tuner adjusts after each one.
        progress: Bool. Shows a progress bar.
        compute_rate: Function from acceptance rate to scale factor.

    # Returns
        Tuner configuration for `model.configure(tuner=...)`. After
        inference `posterior.config["sigma"]` holds the tuned `sigma`,
        in the form it was given.
    """
    args = (sigma, num_steps, num_episodes, progress, compute_rate)
    return TunerConfig(*args, keep_sigma)


def AdaptiveDiagonalTuner(
    sigma, num_steps=200, num_episodes=10, progress=True, compute_rate=None
):
    """Tunes one proposal step per parameter for Metropolis-Hastings.

    The proposal is `scale * sigma`. The scale follows the acceptance rate.
    `sigma` becomes the standard deviation of each parameter, measured
    inside each chain and averaged over chains.

    # Arguments
        sigma: Initial proposal sigma. A number, one value per parameter,
            or a Cholesky factor of the proposal covariance.
        num_steps: Int. Steps per chain in each episode.
        num_episodes: Int. Episodes; the tuner adjusts after each one.
        progress: Bool. Shows a progress bar.
        compute_rate: Function from acceptance rate to scale factor.

    # Returns
        Tuner configuration for `model.configure(tuner=...)`. After
        inference `posterior.config["sigma"]` holds the tuned vector.
    """
    args = (sigma, num_steps, num_episodes, progress, compute_rate)
    return TunerConfig(*args, compute_diagonal_sigma)


def AdaptiveCovarianceTuner(
    sigma, num_steps=200, num_episodes=10, progress=True, compute_rate=None
):
    """Tunes the full proposal covariance for Metropolis-Hastings.

    The proposal is `scale * sigma`. The scale follows the acceptance rate.
    `sigma` becomes the Cholesky factor of the covariance, measured
    inside each chain and averaged over chains.

    # Arguments
        sigma: Initial proposal sigma. A number, one value per parameter,
            or a Cholesky factor of the proposal covariance.
        num_steps: Int. Steps per chain in each episode.
        num_episodes: Int. Episodes; the tuner adjusts after each one.
        progress: Bool. Shows a progress bar.
        compute_rate: Function from acceptance rate to scale factor.

    # Returns
        Tuner configuration for `model.configure(tuner=...)`. After
        inference `posterior.config["sigma"]` holds the tuned Cholesky
        factor.
    """
    args = (sigma, num_steps, num_episodes, progress, compute_rate)
    return TunerConfig(*args, compute_cholesky_sigma)


def keep_sigma(initial_sigma, measured_covariance):
    return initial_sigma


def compute_diagonal_sigma(initial_sigma, measured_covariance):
    return jp.sqrt(jp.diagonal(measured_covariance))


def compute_cholesky_sigma(initial_sigma, measured_covariance):
    return jp.linalg.cholesky(measured_covariance)


def Tuner(
    log_density_fn,
    samples,
    compute_rate=None,
    progress=True,
    compute_sigma=None,
):
    if compute_rate is None:
        compute_rate = AcceptanceToVariance(ACCEPTANCE_RATES, VARIANCE_FACTORS)
    if compute_sigma is None:
        compute_sigma = keep_sigma

    @partial(jax.jit, static_argnums=(1, 2))
    def tune(key, num_steps, num_episodes, initial_sigma):
        initial_sigma = jp.asarray(initial_sigma)
        sigma = compute_first_sigma(compute_sigma, initial_sigma, samples)
        state = build_episode_state(key, log_density_fn, samples, sigma)
        tune_one_chain = partial(tune_chain, log_density_fn, compute_rate)
        episode_args = (tune_one_chain, compute_sigma, initial_sigma, num_steps)
        show_progress = build_progress_callback(progress, num_episodes)

        def step(state, episode):
            show_progress(episode + 1)
            return run_episode(*episode_args, state)

        indices = jp.arange(num_episodes)
        state, episodes = jax.lax.scan(step, state, indices)
        if progress:
            progressbar.newline()
        return build_result(state, episodes)

    return tune


def AcceptanceToVariance(acceptance_rates, variance_factors):
    coefficients = jp.polyfit(acceptance_rates, variance_factors, deg=5)
    return lambda acceptance_rate: jp.polyval(coefficients, acceptance_rate)


def compute_first_sigma(compute_sigma, initial_sigma, samples):
    size = jax.vmap(flatten)(samples).shape[1]
    return compute_sigma(initial_sigma, square_sigma(initial_sigma, size))


def flatten(position):
    return flatten_util.ravel_pytree(position)[0]


def square_sigma(sigma, size):
    if sigma.ndim == 2:
        covariance = sigma @ sigma.T
    else:
        covariance = jp.diag(jp.broadcast_to(sigma**2, (size,)))
    return covariance


def build_episode_state(key, log_density_fn, samples, sigma):
    log_densities = jax.vmap(log_density_fn)(samples)
    ones = jp.ones(log_densities.shape[0])
    tuner_state = TunerState(Samples(samples, log_densities), ones, ones)
    return EpisodeState(key, tuner_state, sigma)


def build_progress_callback(progress, num_episodes):
    if progress:
        callback = progressbar.show(num_episodes, "tuning", width=30)
    else:
        callback = skip_progress
    return callback


def skip_progress(episode):
    return None


def run_episode(tune_one_chain, compute_sigma, initial_sigma, num_steps, state):
    new_key, key = jax.random.split(state.key)
    keys = jax.random.split(key, state.tuner_state.scale.shape[0])
    tune_chains = jax.vmap(partial(tune_one_chain, num_steps, state.sigma))
    tuner_state, chains = tune_chains(state.tuner_state, keys)
    covariance = compute_covariance(chains.positions)
    measured_sigma = compute_sigma(initial_sigma, covariance)
    is_valid = is_valid_sigma(measured_sigma)
    sigma = jp.where(is_valid, measured_sigma, state.sigma)
    new_state = EpisodeState(new_key, tuner_state, sigma)
    return new_state, Episode(chains.info, state.sigma, is_valid)


def tune_chain(log_density_fn, compute_rate, num_steps, sigma, state, key):
    scale = state.factor * state.scale
    step = partial(step_kernel, log_density_fn, scale * sigma)
    keys = jax.random.split(key, num_steps)
    kernel_state, infos = jax.lax.scan(step, state.kernel_state, keys)
    acceptance_rate = jp.mean(infos.is_accepted, axis=0)
    factor = compute_rate(acceptance_rate)
    positions = jax.vmap(flatten)(infos.proposal.state.position)
    info = TunerInfo(scale, factor, acceptance_rate)
    return TunerState(kernel_state, scale, factor), Chain(info, positions)


def step_kernel(log_density_fn, sigma, kernel_state, key):
    key_propose, key_accept = jax.random.split(key)
    position = propose_additively(key_propose, kernel_state.position, sigma)
    new_state = Samples(position, log_density_fn(position))
    now_proposal = build_now_proposal(kernel_state)
    new_proposal = build_new_proposal(kernel_state, new_state)
    info = choose_proposal(key_accept, now_proposal, new_proposal)
    return info.proposal.state, info


def compute_covariance(positions):
    # Centering each chain leaves out the distance between chains.
    num_steps, size = positions.shape[1:]
    centered = positions - positions.mean(axis=1, keepdims=True)
    covariance = jp.einsum("csi,csj->ij", centered, centered)
    covariance = covariance / (positions.shape[0] * num_steps)
    # Stan's shrinkage keeps the matrix positive definite with few samples.
    # It is relative because posteriors here are often far narrower than 1.
    shrinkage = 1e-3 * 5 / (num_steps + 5) * jp.mean(jp.diagonal(covariance))
    weight = num_steps / (num_steps + 5)
    return weight * covariance + shrinkage * jp.eye(size)


def is_valid_sigma(sigma):
    # Chains that never moved, or a covariance that is singular in float32,
    # give no usable sigma.
    return jp.all(jp.isfinite(sigma)) & jp.any(sigma != 0)


def build_result(state, episodes):
    infos = episodes.infos
    # The last factor was measured with the last sigma that was used.
    scale = (infos.scale[-1] * infos.factor[-1]).mean()
    positions = state.tuner_state.kernel_state.position
    args = (scale * episodes.sigma[-1], positions, infos)
    return TunerResult(*args, episodes.is_sigma_valid)
