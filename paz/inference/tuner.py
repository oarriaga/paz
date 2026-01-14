from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp

from .metropolis_hastings import (
    Proposal,
    Samples,
    apply_gaussian_noise,
    choose_proposal,
    symmetric_proposal_log_density_fn,
)


TunerInfo = namedtuple("TunerInfo", ["sigma", "factor", "acceptance_rate"])
TunerState = namedtuple("TunerState", ["kernel_state", "sigma", "factor"])
ACCEPTANCE_RATES = jp.array([0.001, 0.05, 0.2, 0.5, 0.75, 0.95])
VARIANCE_FACTORS = jp.array([0.1, 0.5, 0.9, 1.1, 2, 10])


def Tuner(log_density_fn, samples, num_chains, compute_rate=None):
    if compute_rate is None:
        compute_rate = AcceptanceToVariance(ACCEPTANCE_RATES, VARIANCE_FACTORS)

    def tune_episode(tuner_state, key, num_steps):
        sigma = tuner_state.factor * tuner_state.sigma
        propose = partial(propose_with_sigma, sigma=sigma)

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
        acceptance_rate = compute_acceptance_rate(infos)
        rate = compute_rate(acceptance_rate)
        new_state = TunerState(kernel_state, sigma, rate)
        return new_state, TunerInfo(sigma, rate, acceptance_rate)

    @partial(jax.jit, static_argnums=(1, 2))
    def tune(key, num_steps, num_episodes, sigma):

        def one_episode(episode_state, sample_arg):
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
        tuned_sigma = (infos.sigma[-1] * infos.factor[-1]).mean()
        return tuned_sigma, infos

    return tune


def AcceptanceToVariance(acceptance_rates, variance_factors):
    coefficients = jp.polyfit(acceptance_rates, variance_factors, deg=5)

    def apply(acceptance_rate):
        variance_factor = jp.polyval(coefficients, acceptance_rate)
        return variance_factor

    return apply


def compute_acceptance_rate(infos):
    return jp.mean(infos.is_accepted, axis=0)


def propose_with_sigma(key, position, sigma):
    move_proposal = apply_gaussian_noise(key, position, 0.0, sigma)
    new_position = jax.tree_util.tree_map(jp.add, position, move_proposal)
    return new_position


def build_now_proposal(state):
    return Proposal(state, 0.0, 0.0, -jp.inf)


def build_new_proposal(old_state, new_state):
    new_energy = symmetric_proposal_log_density_fn(old_state, new_state)
    old_energy = symmetric_proposal_log_density_fn(new_state, old_state)
    delta_energy = old_energy - new_energy
    sum_log_p_accept = jp.minimum(delta_energy, 0.0)
    return Proposal(new_state, new_energy, delta_energy, sum_log_p_accept)
