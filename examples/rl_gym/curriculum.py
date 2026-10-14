import jax.numpy as jp


def update_max_speed(max_speed, tracking, episode_length, iteration, num_steps=24, episode_steps=1000, increase=0.1, threshold=0.8):  # fmt: skip
    # the reference curriculum rates the episodic tracking return against
    # the full episode horizon, so commands only widen once the policy
    # both tracks well and survives whole episodes
    period = iteration * num_steps // episode_steps
    previous = (iteration - 1) * num_steps // episode_steps
    survival = jp.minimum(episode_length / episode_steps, 1.0)
    earned = (period > previous) & (tracking * survival > threshold)
    return jp.minimum(max_speed + jp.where(earned, increase, 0.0), 1.0)
