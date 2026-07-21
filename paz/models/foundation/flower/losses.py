"""Rectified-flow training targets and loss.

The forward interpolation is ``(1 - t) * actions + t * noise`` so t=0 is
data and t=1 is noise; the regression target is ``noise - actions``.
"""
import jax.numpy as jp


def interpolate_actions(actions, noise, times):
    times = times[:, None, None]
    return (1.0 - times) * actions + times * noise


def build_target_velocity(actions, noise):
    return noise - actions


def rectified_flow_loss(velocity, target_velocity, mask):
    squared_error = (target_velocity - velocity) ** 2
    weights = jp.broadcast_to(mask[..., None], squared_error.shape)
    return jp.sum(weights * squared_error) / jp.sum(weights)
