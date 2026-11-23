from collections import namedtuple

import jax
from jax import flatten_util
import jax.numpy as jp

import paz.utils.progressbar as progressbar

Samples = namedtuple("Samples", ["position", "log_density"])
State = namedtuple("State", ["proposal", "is_accepted", "acceptance_rate"])
Proposal = namedtuple(
    "Proposal", ["state", "energy", "weight", "sum_log_p_accept"]
)


def apply_gaussian_noise(key, position, mu=0.0, sigma=1.0):
    flat_pytree, unravel = flatten_util.ravel_pytree(position)
    flat_shape = flat_pytree.shape
    flat_dtype = flat_pytree.dtype
    sample = jax.random.normal(key, shape=flat_shape, dtype=flat_dtype)
    dtype = jp.result_type(sigma, sample)
    sigma = jp.asarray(sigma, dtype=dtype)
    sample = sample.astype(dtype)
    if jp.ndim(sigma) <= 1:
        mapped = jax.lax.mul(sigma, sample)
    else:
        mapped = jax.lax.dot(sigma, sample, precision="highest")
    return unravel(mu + mapped)


def choose_proposal(key, now_proposal, new_proposal):
    acceptance_probability = jp.clip(jp.exp(new_proposal.weight), max=1)
    do_accept = jax.random.bernoulli(key, acceptance_probability)
    return jax.lax.cond(
        do_accept,
        lambda _: State(new_proposal, do_accept, acceptance_probability),
        lambda _: State(now_proposal, do_accept, acceptance_probability),
        operand=None,
    )


def build_now_proposal(state):
    return Proposal(state, 0.0, 0.0, -jp.inf)


def build_new_proposal(old_state, new_state):
    new_energy = -new_state.log_density
    delta_energy = new_state.log_density - old_state.log_density
    sum_log_p_accept = jp.minimum(delta_energy, 0.0)
    return Proposal(new_state, new_energy, delta_energy, sum_log_p_accept)


def propose_additively(key, position, sigma):
    move_proposal = apply_gaussian_noise(key, position, 0.0, sigma)
    new_position = jax.tree_util.tree_map(jp.add, position, move_proposal)
    return new_position


def sample(
    key,
    log_density_fn,
    positions,
    sigma,
    num_samples,
    num_chains,
    show_progress=progressbar.skip,
    callbacks=(),
):
    def build_trajectory(key, initial_state):
        position, log_density = initial_state
        new_position = propose_additively(key, position, sigma)
        return Samples(new_position, log_density_fn(new_position))

    def step_kernel(key, state):
        keys = jax.random.split(key)
        end_state = build_trajectory(keys[0], state)
        now_proposal = build_now_proposal(state)
        new_proposal = build_new_proposal(state, end_state)
        sample = choose_proposal(keys[1], now_proposal, new_proposal)
        return sample.proposal.state, sample

    def step_chain(step_state, sample_arg):
        show_progress(sample_arg + 1)
        old_key, states = step_state
        new_key, now_key = jax.random.split(old_key)
        keys = jax.random.split(now_key, num_chains)
        states, infos = jax.vmap(step_kernel)(keys, states)
        callback_args = (sample_arg, states, infos.acceptance_rate)
        run_callbacks(callbacks, "sample", *callback_args)
        return (new_key, states), (states, infos)

    args = jp.arange(num_samples)
    state = Samples(positions, jax.vmap(log_density_fn)(positions))
    _, (states, infos) = jax.lax.scan(step_chain, (key, state), args)
    return states, infos


def run_callbacks(callbacks, stage, step_arg, samples, acceptance_rate):
    """Calls each host function with the state of every chain, as
    `callback(stage, step_arg, samples, acceptance_rate)`.

    In stage "tune" a step is one episode and the rate is the share of
    accepted moves in it. In stage "sample" a step is one draw, warmup
    included, and the rate is the acceptance probability of that draw.
    """
    if len(callbacks) > 0:

        def observe(step_arg, samples, acceptance_rate):
            for callback in callbacks:
                callback(stage, step_arg, samples, acceptance_rate)

        args = (step_arg, samples, acceptance_rate)
        jax.debug.callback(observe, *args, ordered=True)
