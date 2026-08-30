from collections import namedtuple
from functools import partial

import optax
import jax
import paz
import jax.numpy as jp
from jax import random as jr

from networks import call_actor
from networks import call_critic
from networks import pack_parameters
from networks import read_stdv
from networks import unpack_parameters

Experience = namedtuple("Experience", "actor_observation, critic_observation, action, log_probability, mean, stdv, value, value_target, advantage, valid")  # fmt: skip
TrainingState = namedtuple("TrainingState", "parameters, optimizer_state, learning_rate")  # fmt: skip
LossTerms = namedtuple("LossTerms", "policy_loss, value_loss, entropy, KL")
UpdateMetrics = namedtuple("UpdateMetrics", "loss, policy_loss, value_loss, entropy, KL, gradient_norm, learning_rate")  # fmt: skip


def build_update(actor, critic, optimizer, num_shards=1, num_epochs=5, num_batches=4, entropy_weight=0.01):  # fmt: skip

    def update(seed, state, experience):
        run_batch = partial(apply_batch, actor, critic, optimizer, entropy_weight)  # fmt: skip
        experience = split_shards(experience, num_shards)

        def run_epoch(state, key):
            shuffled = shuffle_shards(key, experience)
            batches = split_batches(shuffled, num_batches)
            return jax.lax.scan(run_batch, state, batches)

        keys = jr.split(jr.key(seed), num_epochs)
        state, metrics = jax.lax.scan(run_epoch, state, keys)
        return state, jax.tree.map(jp.mean, metrics)

    return jax.jit(update)


def split_shards(experience, num_shards):
    # each shard is one device's rollout; shuffling and minibatching stay
    # inside a shard so a multi-process update never gathers across devices

    def split(values):
        return values.reshape((num_shards, -1) + values.shape[1:])

    return jax.tree.map(split, experience)


def shuffle_shards(key, experience):
    keys = jr.split(key, count_samples(experience))
    return jax.vmap(shuffle_experience)(keys, experience)


def shuffle_experience(key, experience):
    indices = jr.permutation(key, count_samples(experience))
    return jax.tree.map(lambda values: values[indices], experience)


def split_batches(experience, num_batches):

    def split(values):
        shape = (values.shape[0], num_batches, -1) + values.shape[2:]
        return jp.moveaxis(values.reshape(shape), 1, 0)

    return jax.tree.map(split, experience)


def count_samples(experience):
    any_field = jax.tree.leaves(experience)[0]
    return any_field.shape[0]


def apply_batch(actor, critic, optimizer, entropy_weight, state, batch):
    batch = jax.tree.map(merge_shards, batch)
    variables = pack_parameters(state.parameters)
    compute_grads = jax.value_and_grad(compute_loss, 2, has_aux=True)
    grad_args = actor, critic, variables, batch
    (loss, metrics), gradients = compute_grads(*grad_args, entropy_weight=entropy_weight)  # fmt: skip
    learning_rate = adapt_learning_rate(state.learning_rate, metrics.KL)
    optimizer_state = set_learning_rate(state.optimizer_state, learning_rate)
    gradients, gradient_norm = clip_gradients(gradients)
    apply_args = optimizer_state, gradients, variables
    variables, optimizer_state = optimizer.stateless_apply(*apply_args)
    parameters = unpack_parameters(variables)
    state = TrainingState(parameters, optimizer_state, learning_rate)
    reported = loss, *metrics, gradient_norm, learning_rate
    return state, UpdateMetrics(*reported)


def adapt_learning_rate(learning_rate, KL, desired_KL=0.01, minimum_rate=1e-5, maximum_rate=1e-2, step=1.5):  # fmt: skip
    too_large = KL > (2.0 * desired_KL)
    too_small = (KL < 0.5 * desired_KL) & (KL > 0.0)
    decreased, increased = learning_rate / step, learning_rate * step
    adapted = jp.where(too_large, decreased, learning_rate)
    adapted = jp.where(too_small, increased, adapted)
    return jp.clip(adapted, minimum_rate, maximum_rate)


def set_learning_rate(optimizer_state, learning_rate, rate_slot=1):
    rate = jp.asarray(learning_rate, dtype=optimizer_state[rate_slot].dtype)
    values = list(optimizer_state)
    values[rate_slot] = rate
    return values


def merge_shards(values):
    return values.reshape((-1,) + values.shape[2:])


def clip_gradients(gradients, max_gradient_norm=1.0, epsilon=1e-6):
    norm = optax.tree.norm(gradients)
    scale = jp.minimum(1.0, max_gradient_norm / (norm + epsilon))
    gradients = jax.tree.map(lambda gradient: gradient * scale, gradients)
    return gradients, norm


def compute_loss(actor, critic, variables, batch, clip_ratio=0.2, value_weight=1.0, entropy_weight=0.01):  # fmt: skip
    parameters = unpack_parameters(variables)
    stdv = read_stdv(parameters)
    mean = call_actor(actor, parameters.actor, batch.actor_observation)
    values = call_critic(critic, parameters.critic, batch.critic_observation)
    log_prob = compute_normal_logprob(batch.action, mean, stdv)
    policy_loss = compute_policy_loss(log_prob, batch, clip_ratio)
    value_loss = compute_value_loss(values, batch, clip_ratio)
    entropy = jp.mean(normal_entropy(stdv))
    weighted_value = value_weight * value_loss
    loss = policy_loss + weighted_value - (entropy_weight * entropy)
    KL = compute_KL(mean, stdv, batch.mean, batch.stdv)
    return loss, LossTerms(policy_loss, value_loss, entropy, KL)


def compute_policy_loss(log_prob, batch, clip_ratio):
    ratio = compute_likelihood_ratio(log_prob, batch.log_probability)
    clipped_ratio = jp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    loss = -compute_surrogate_objective(ratio, batch.advantage)
    clipped_loss = -compute_surrogate_objective(clipped_ratio, batch.advantage)
    return compute_pessimistic_loss(loss, clipped_loss, batch.valid)


def compute_likelihood_ratio(log_prob, old_log_prob, bound=20.0):
    # the clip keeps a corrupted sample from overflowing exp into inf and
    # poisoning the gradients; it is far outside the trust region, so it
    # never changes a healthy update
    return jp.exp(jp.clip(log_prob - old_log_prob, -bound, bound))


def compute_surrogate_objective(likelihood_ratio, advantage):
    return likelihood_ratio * advantage


def compute_value_loss(values, batch, clip_ratio):
    lower, upper = batch.value - clip_ratio, batch.value + clip_ratio
    clipped_values = jp.clip(values, lower, upper)
    loss = paz.losses.mse(batch.value_target, values, reduction="none")
    clipped_loss = paz.losses.mse(batch.value_target, clipped_values, reduction="none")  # fmt: skip
    return compute_pessimistic_loss(loss, clipped_loss, batch.valid)


def compute_pessimistic_loss(loss, clipped_loss, valid):
    # the larger of the two is the less favourable reading, so an update can
    # never look cheap only because the policy moved far from the one that
    # collected the batch; diverged steps carry no information and are
    # masked out of the mean
    pessimistic = jp.maximum(loss, clipped_loss) * valid
    return jp.sum(pessimistic) / jp.maximum(jp.sum(valid), 1.0)


def compute_KL(mean, stdv, old_mean, old_stdv, epsilon=1e-5):
    mean_error = paz.losses.mse(old_mean, mean, reduction="none")
    variance = old_stdv**2 + mean_error
    ratio = jp.log(stdv / old_stdv + epsilon)
    terms = ratio + variance / (2.0 * stdv**2) - 0.5
    return jp.mean(jp.sum(terms, axis=-1))


def normal_entropy(stdv):
    return jp.sum(jp.log(stdv) + 0.5 + 0.5 * jp.log(2.0 * jp.pi), axis=-1)


def compute_value_targets(rewards, dones, values, last_values, gamma=0.99, GAE_lambda=0.95):  # fmt: skip

    def accumulate(carry, inputs):
        next_values, advantage = carry
        reward, done, value = inputs
        not_done = 1.0 - done
        delta = reward + (gamma * next_values * not_done) - value
        advantage = delta + (gamma * GAE_lambda * not_done * advantage)
        return (value, advantage), advantage + value

    carry = (last_values, jp.zeros_like(last_values))
    reversed_inputs = rewards[::-1], dones[::-1], values[::-1]
    _, returns = jax.lax.scan(accumulate, carry, reversed_inputs)
    return returns[::-1]


def sample_actions(key, mean, stdv):
    noise = jax.random.normal(key, mean.shape)
    actions = mean + noise * stdv
    return actions, compute_normal_logprob(actions, mean, stdv)


def compute_normal_logprob(actions, mean, stdv):
    log_probabilities = jax.scipy.stats.norm.logpdf(actions, mean, stdv)
    return jp.sum(log_probabilities, axis=-1)


def standardize_advantages(advantages, valid, epsilon=1e-8):
    # diverged samples are masked out of the statistics so a single
    # corrupted advantage cannot flatten every other sample's signal
    count = jp.maximum(jp.sum(valid), 2.0)
    mean = jp.sum(advantages * valid) / count
    squared = ((advantages - mean) ** 2) * valid
    stdv = jp.sqrt(jp.sum(squared) / (count - 1.0))
    return (advantages - mean) / (stdv + epsilon)
