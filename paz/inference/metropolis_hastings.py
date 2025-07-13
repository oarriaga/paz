from functools import partial
from collections import namedtuple

import jax
from jax import flatten_util
import jax.numpy as jp


Samples = namedtuple("Samples", ["position", "log_density"])
State = namedtuple("State", ["proposal", "is_accepted", "acceptance_rate"])
Proposal = namedtuple(
    "Proposal", ["state", "energy", "weight", "sum_log_p_accept"]
)


@partial(jax.jit, static_argnames=("precision",), inline=True)
def linear_map(diagonal_or_dense_A, x, *, precision="highest"):
    """Perform a linear map of the form y = Ax."""
    dtype = jp.result_type(diagonal_or_dense_A.dtype, x.dtype)
    diagonal_or_dense_A = diagonal_or_dense_A.astype(dtype)
    x = x.astype(dtype)
    ndim = jp.ndim(diagonal_or_dense_A)

    if ndim <= 1:
        return jax.lax.mul(diagonal_or_dense_A, x)
    else:
        return jax.lax.dot(diagonal_or_dense_A, x, precision=precision)


def apply_gaussian_noise(key, position, mu=0.0, sigma=1.0):
    """Generate N(mu, sigma) noise with structure that matches a PyTree."""
    flat_pytree, unravel = flatten_util.ravel_pytree(position)
    flat_shape = flat_pytree.shape
    flat_dtype = flat_pytree.dtype
    sample = jax.random.normal(key, shape=flat_shape, dtype=flat_dtype)
    return unravel(mu + linear_map(sigma, sample))


def choose_proposal(key, now_proposal, new_proposal):
    acceptance_probability = jp.clip(jp.exp(new_proposal.weight), a_max=1)
    do_accept = jax.random.bernoulli(key, acceptance_probability)
    return jax.lax.cond(
        do_accept,
        lambda _: State(new_proposal, do_accept, acceptance_probability),
        lambda _: State(now_proposal, do_accept, acceptance_probability),
        operand=None,
    )


def symmetric_proposal_log_density_fn(old_state, new_state):
    return -new_state.log_density


def propose_additively(key, position, sigma):
    move_proposal = apply_gaussian_noise(key, position, 0.0, sigma)
    new_position = jax.tree_util.tree_map(jp.add, position, move_proposal)
    return new_position


def sample(key, log_density_fn, positions, sigma, num_samples, num_chains):

    def build_trajectory(key, initial_state):
        position, log_density = initial_state
        new_position = propose_additively(key, position, sigma)
        return Samples(new_position, log_density_fn(new_position))

    def build_now_proposal(state):
        return Proposal(state, 0.0, 0.0, -jp.inf)

    def build_new_proposal(old_state, new_state):
        new_energy = symmetric_proposal_log_density_fn(old_state, new_state)
        old_energy = symmetric_proposal_log_density_fn(new_state, old_state)
        delta_energy = old_energy - new_energy
        sum_log_p_accept = jp.minimum(delta_energy, 0.0)
        return Proposal(new_state, new_energy, delta_energy, sum_log_p_accept)

    def step_kernel(key, state):
        keys = jax.random.split(key)
        end_state = build_trajectory(keys[0], state)
        now_proposal = build_now_proposal(state)
        new_proposal = build_new_proposal(state, end_state)
        sample = choose_proposal(keys[1], now_proposal, new_proposal)
        return sample.proposal.state, sample

    def step_chain(step_state, sample_arg):
        old_key, states = step_state
        new_key, now_key = jax.random.split(old_key)
        keys = jax.random.split(now_key, num_chains)
        states, infos = jax.vmap(step_kernel)(keys, states)
        return (new_key, states), (states, infos)

    args = jp.arange(num_samples)
    state = Samples(positions, jax.vmap(log_density_fn)(positions))
    _, (states, infos) = jax.lax.scan(step_chain, (key, state), args)
    return states, infos
