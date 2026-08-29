import jax.numpy as jp


def update_max_speed(max_speed, tracking, iteration, num_steps=24, episode_steps=1000, increase=0.1, threshold=0.8):  # fmt: skip
    period = iteration * num_steps // episode_steps
    previous = (iteration - 1) * num_steps // episode_steps
    earned = (period > previous) & (tracking > threshold)
    return jp.minimum(max_speed + jp.where(earned, increase, 0.0), 1.0)
