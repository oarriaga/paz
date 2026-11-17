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
ProposalShape = namedtuple("ProposalShape", ["build", "update"])


def build_tuner_type(name):
    fields = ["sigma", "num_steps", "num_episodes", "progress", "compute_rate"]
    return namedtuple(name, fields, defaults=[200, 10, True, None])


AdaptiveStepTuner = build_tuner_type("AdaptiveStepTuner")
AdaptiveDiagonalTuner = build_tuner_type("AdaptiveDiagonalTuner")
AdaptiveCovarianceTuner = build_tuner_type("AdaptiveCovarianceTuner")
ACCEPTANCE_RATES = jp.array([0.001, 0.05, 0.2, 0.5, 0.75, 0.95])
VARIANCE_FACTORS = jp.array([0.1, 0.5, 0.9, 1.1, 2, 10])


def Tuner(
    log_density_fn,
    samples,
    num_chains,
    compute_rate=None,
    progress=True,
    proposal_shape=None,
):
    if compute_rate is None:
        compute_rate = AcceptanceToVariance(ACCEPTANCE_RATES, VARIANCE_FACTORS)
    if proposal_shape is None:
        proposal_shape = PROPOSAL_SHAPES[AdaptiveStepTuner]

    def tune_episode(tuner_state, key, shape, num_steps):
        scale = tuner_state.factor * tuner_state.scale
        propose = partial(propose_additively, sigma=scale * shape)

        def kernel_step(kernel_state, key):
            key_propose, key_accept = jax.random.split(key)
            new_position = propose(key_propose, kernel_state.position)
            new_state = Samples(new_position, log_density_fn(new_position))
            now_proposal = build_now_proposal(kernel_state)
            new_proposal = build_new_proposal(kernel_state, new_state)
            info = choose_proposal(key_accept, now_proposal, new_proposal)
            return info.proposal.state, info

        keys = jax.random.split(key, num_steps)
        kernel_state, infos = jax.lax.scan(
            kernel_step, tuner_state.kernel_state, keys
        )
        acceptance_rate = jp.mean(infos.is_accepted, axis=0)
        rate = compute_rate(acceptance_rate)
        new_state = TunerState(kernel_state, scale, rate)
        positions = jax.vmap(flatten)(infos.proposal.state.position)
        return new_state, (TunerInfo(scale, rate, acceptance_rate), positions)

    @partial(jax.jit, static_argnums=(1, 2))
    def tune(key, num_steps, num_episodes, sigma):
        if progress:
            progress_callback = progressbar.show(
                num_episodes, "tuning", width=30
            )
        else:
            progress_callback = None

        def one_episode(episode_state, sample_arg):
            if progress:
                progress_callback(sample_arg + 1)
            now_key, tune_states, shape = episode_state
            new_key, key = jax.random.split(now_key)
            keys = jax.random.split(key, num_chains)
            tune_step = partial(tune_episode, shape=shape, num_steps=num_steps)
            outputs = jax.vmap(tune_step)(tune_states, keys)
            tune_states, (infos, positions) = outputs
            new_shape = proposal_shape.update(positions, shape)
            # Chains that never moved, or a covariance that is singular in
            # float32, give no usable shape.
            is_valid = jp.all(jp.isfinite(new_shape)) & jp.any(new_shape != 0)
            new_shape = jp.where(is_valid, new_shape, shape)
            return (new_key, tune_states, new_shape), (infos, shape)

        def build_kernel_state(position):
            return Samples(position, log_density_fn(position))

        kernel_state = jax.vmap(build_kernel_state)(samples)
        ones = jp.ones(num_chains)
        tuner_state = TunerState(kernel_state, ones, ones)
        size = jax.vmap(flatten)(samples).shape[1]
        episode_state = (key, tuner_state, proposal_shape.build(sigma, size))
        episode_args = jp.arange(num_episodes)
        outputs = jax.lax.scan(one_episode, episode_state, episode_args)
        (_, tuner_state, _), (infos, shapes) = outputs
        if progress:
            progressbar.newline()
        # The last factor was measured with the last shape that was used.
        tuned_scale = (infos.scale[-1] * infos.factor[-1]).mean()
        positions = tuner_state.kernel_state.position
        return tuned_scale * shapes[-1], positions, infos

    return tune


def flatten(position):
    return flatten_util.ravel_pytree(position)[0]


def build_scalar_shape(sigma, size):
    return jp.asarray(sigma, dtype=float)


def keep_shape(positions, shape):
    return shape


def build_diagonal_shape(sigma, size):
    return jp.full(size, sigma, dtype=float)


def update_diagonal_shape(positions, shape):
    return jp.sqrt(jp.diagonal(compute_covariance(positions)))


def build_cholesky_shape(sigma, size):
    return sigma * jp.eye(size)


def update_cholesky_shape(positions, shape):
    return jp.linalg.cholesky(compute_covariance(positions))


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


scalar_shape = ProposalShape(build_scalar_shape, keep_shape)
diagonal_shape = ProposalShape(build_diagonal_shape, update_diagonal_shape)
cholesky_shape = ProposalShape(build_cholesky_shape, update_cholesky_shape)
PROPOSAL_SHAPES = {
    AdaptiveStepTuner: scalar_shape,
    AdaptiveDiagonalTuner: diagonal_shape,
    AdaptiveCovarianceTuner: cholesky_shape,
}


def AcceptanceToVariance(acceptance_rates, variance_factors):
    coefficients = jp.polyfit(acceptance_rates, variance_factors, deg=5)
    return lambda acceptance_rate: jp.polyval(coefficients, acceptance_rate)
