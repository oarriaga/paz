import jax.numpy as jp


def update_max_speed(max_speed, episodic_tracking, iteration, num_steps=24, episode_steps=1000, increase=0.1, threshold=0.8):  # fmt: skip
    # the reference widens the commands once per episode horizon when the
    # episodes ending there earned 80% of the full tracking return, so the
    # policy must both track well and survive whole episodes
    period = iteration * num_steps // episode_steps
    previous = (iteration - 1) * num_steps // episode_steps
    earned = (period > previous) & (episodic_tracking > threshold)
    return jp.minimum(max_speed + jp.where(earned, increase, 0.0), 1.0)
