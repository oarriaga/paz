import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax.numpy as jp
import numpy as np
from jax import random as jr

import ppo


def reference_value_targets(rewards, dones, values, last_values, gamma, lam):
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    advantage = np.zeros_like(last_values)
    next_values = last_values
    for step in reversed(range(num_steps)):
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        advantage = delta + gamma * lam * not_done * advantage
        advantages[step] = advantage
        next_values = values[step]
    return advantages + values


def test_value_targets_match_reference():
    keys = jr.split(jr.key(0), 3)
    rewards = jr.normal(keys[0], (7, 5))
    dones = (jr.uniform(keys[1], (7, 5)) < 0.3).astype(jp.float32)
    values = jr.normal(keys[2], (7, 5))
    last_values = jp.zeros(5)
    targets = ppo.compute_value_targets(rewards, dones, values, last_values)
    args = map(np.asarray, (rewards, dones, values, last_values))
    reference = reference_value_targets(*args, 0.99, 0.95)
    assert np.allclose(np.asarray(targets), reference, atol=1e-5)


def test_adapt_learning_rate():
    rate = jp.asarray(1e-3)
    decreased = ppo.adapt_learning_rate(rate, jp.asarray(0.05))
    assert np.isclose(float(decreased), 1e-3 / 1.5)
    increased = ppo.adapt_learning_rate(rate, jp.asarray(0.001))
    assert np.isclose(float(increased), 1e-3 * 1.5)
    unchanged = ppo.adapt_learning_rate(rate, jp.asarray(0.01))
    assert np.isclose(float(unchanged), 1e-3)
    floor = ppo.adapt_learning_rate(jp.asarray(1e-5), jp.asarray(1.0))
    assert np.isclose(float(floor), 1e-5)
    ceiling = ppo.adapt_learning_rate(jp.asarray(1e-2), jp.asarray(0.0))
    assert np.isclose(float(ceiling), 1e-2)


def test_KL_matches_analytic_gaussian():
    keys = jr.split(jr.key(1), 4)
    mean = jr.normal(keys[0], (6, 3))
    old_mean = jr.normal(keys[1], (6, 3))
    stdv = jp.exp(jr.normal(keys[2], (3,)) * 0.1)
    old_stdv = jp.exp(jr.normal(keys[3], (6, 3)) * 0.1)
    computed = float(ppo.compute_KL(mean, stdv, old_mean, old_stdv))
    ratio = np.log(np.asarray(stdv) / np.asarray(old_stdv))
    squared = (np.asarray(old_mean) - np.asarray(mean)) ** 2
    variance = np.asarray(old_stdv) ** 2 + squared
    terms = ratio + variance / (2.0 * np.asarray(stdv) ** 2) - 0.5
    reference = float(np.mean(np.sum(terms, axis=-1)))
    assert np.isclose(computed, reference, atol=1e-3)


def test_logprob_matches_gaussian_density():
    mean, stdv = jp.array([0.5, -1.0]), jp.array([0.3, 2.0])
    action = jp.array([0.7, 0.0])
    computed = float(ppo.compute_normal_logprob(action, mean, stdv))
    densities = np.exp(-0.5 * ((np.asarray(action) - np.asarray(mean)) / np.asarray(stdv)) ** 2)  # fmt: skip
    densities = densities / (np.asarray(stdv) * np.sqrt(2.0 * np.pi))
    assert np.isclose(computed, float(np.sum(np.log(densities))), atol=1e-5)


def test_entropy_matches_gaussian_formula():
    stdv = jp.array([0.5, 1.5])
    computed = float(ppo.normal_entropy(stdv))
    reference = float(np.sum(0.5 * np.log(2.0 * np.pi * np.e * np.asarray(stdv) ** 2)))  # fmt: skip
    assert np.isclose(computed, reference, atol=1e-5)


def test_clip_gradients_scales_large_norms():
    gradients = [jp.full((4,), 3.0)]
    clipped, norm = ppo.clip_gradients(gradients)
    assert np.isclose(float(norm), 6.0)
    assert np.isclose(float(jp.linalg.norm(clipped[0])), 1.0, atol=1e-4)
    small = [jp.full((4,), 0.1)]
    unclipped, _ = ppo.clip_gradients(small)
    assert np.allclose(np.asarray(unclipped[0]), 0.1, atol=1e-5)


def build_batch(size, **fields):
    values = {name: jp.zeros(size) for name in ppo.Experience._fields}
    values.update(fields)
    return ppo.Experience(**values)


def test_policy_loss_clips_large_ratios():
    batch = build_batch(2, log_probability=jp.zeros(2), advantage=jp.ones(2))
    log_prob = jp.log(jp.array([2.0, 1.0]))
    loss = float(ppo.compute_policy_loss(log_prob, batch, clip_ratio=0.2))
    assert np.isclose(loss, -(1.2 + 1.0) / 2.0, atol=1e-5)


def test_value_loss_clips_large_updates():
    batch = build_batch(2, value=jp.zeros(2), value_target=jp.array([1.0, 0.1]))  # fmt: skip
    values = jp.array([0.5, 0.1])
    loss = float(ppo.compute_value_loss(values, batch, clip_ratio=0.2))
    expected = ((1.0 - 0.2) ** 2 + 0.0) / 2.0
    assert np.isclose(loss, expected, atol=1e-5)


def test_standardize_advantages():
    advantages = jr.normal(jr.key(2), (100,)) * 3.0 + 5.0
    standardized = ppo.standardize_advantages(advantages)
    assert np.isclose(float(jp.mean(standardized)), 0.0, atol=1e-5)
    assert np.isclose(float(jp.std(standardized, ddof=1)), 1.0, atol=1e-3)


def test_shuffle_permutes_and_split_batches_partitions():
    experience = build_batch(8, action=jp.arange(8.0))
    shuffled = ppo.shuffle_experience(jr.key(3), experience)
    assert np.allclose(np.sort(np.asarray(shuffled.action)), np.arange(8.0))
    batches = ppo.split_batches(experience, 2)
    assert batches.action.shape == (2, 4)
    assert np.allclose(np.asarray(batches.action[1]), [4.0, 5.0, 6.0, 7.0])
