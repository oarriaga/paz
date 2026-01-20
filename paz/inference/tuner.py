from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp

from paz import progressbar
from .metropolis_hastings import (
    Samples,
    build_new_proposal,
    build_now_proposal,
    choose_proposal,
    propose_additively,
)


TunerInfo = namedtuple("TunerInfo", ["sigma", "factor", "acceptance_rate"])
TunerState = namedtuple("TunerState", ["kernel_state", "sigma", "factor"])
AdaptiveStepTuner = namedtuple(
    "AdaptiveStepTuner",
    ["sigma", "num_steps", "num_episodes", "progress", "compute_rate"],
    defaults=[200, 10, True, None],
)
ACCEPTANCE_RATES = jp.array([0.001, 0.05, 0.2, 0.5, 0.75, 0.95])
VARIANCE_FACTORS = jp.array([0.1, 0.5, 0.9, 1.1, 2, 10])


def Tuner(log_density_fn, samples, num_chains, compute_rate=None, progress=True):
    if compute_rate is None:
        compute_rate = AcceptanceToVariance(ACCEPTANCE_RATES, VARIANCE_FACTORS)

    def tune_episode(tuner_state, key, num_steps):
        sigma = tuner_state.factor * tuner_state.sigma
        propose = partial(propose_additively, sigma=sigma)

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
        new_state = TunerState(kernel_state, sigma, rate)
        return new_state, TunerInfo(sigma, rate, acceptance_rate)

    @partial(jax.jit, static_argnums=(1, 2))
    def tune(key, num_steps, num_episodes, sigma):
        progress_callback = (
            progressbar.show(num_episodes, "tuning", width=30)
            if progress
            else None
        )

        def one_episode(episode_state, sample_arg):
            if progress:
                progress_callback(sample_arg + 1)
            now_key, tune_states = episode_state
            new_key, key = jax.random.split(now_key)
            keys = jax.random.split(key, num_chains)
            tune_step = partial(tune_episode, num_steps=num_steps)
            tune_states, infos = jax.vmap(tune_step)(tune_states, keys)
            return (new_key, tune_states), infos

        sigma = jp.full(num_chains, sigma)
        rates = jp.ones(num_chains)
        kernel_state = jax.vmap(
            lambda position: Samples(position, log_density_fn(position))
        )(samples)
        tuner_state = TunerState(kernel_state, sigma, rates)
        episode_args = jp.arange(num_episodes)
        _, infos = jax.lax.scan(one_episode, (key, tuner_state), episode_args)
        if progress:
            progressbar.newline()
        tuned_sigma = (infos.sigma[-1] * infos.factor[-1]).mean()
        return tuned_sigma, infos

    return tune


def AcceptanceToVariance(acceptance_rates, variance_factors):
    coefficients = jp.polyfit(acceptance_rates, variance_factors, deg=5)
    return lambda acceptance_rate: jp.polyval(coefficients, acceptance_rate)
