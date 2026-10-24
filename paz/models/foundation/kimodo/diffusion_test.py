import jax.numpy as jp
import numpy as np

from paz.models.foundation.kimodo import diffusion


def test_schedule_shapes_and_bounds():
    schedule = diffusion.build_schedule(10)
    assert schedule.base_timesteps.shape == (10,)
    assert int(schedule.base_timesteps[0]) == 0
    assert int(schedule.base_timesteps[-1]) == 999
    assert float(schedule.alphas_prev[0]) == 1.0
    assert bool((jp.diff(schedule.alphas_prev[1:]) <= 0).all())


def test_respaced_timesteps_round_to_base_steps():
    assert diffusion.space_timesteps(3, 1000)[:3].tolist() == [0, 500, 999]
    assert diffusion.space_timesteps(1000, 1000)[:3].tolist() == [0, 1, 2]


def test_full_schedule_matches_base_alphas():
    schedule = diffusion.build_schedule(1000)
    alphas = np.cumprod(1.0 - diffusion.build_cosine_betas(1000))
    recovered = 1.0 / np.asarray(schedule.sqrt_recip_alphas) ** 2
    assert np.allclose(recovered, alphas, rtol=1e-4)


def test_last_step_returns_the_clean_prediction():
    schedule = diffusion.build_schedule(5)
    noisy = jp.ones((1, 2, 3))
    clean = jp.full((1, 2, 3), 0.25)
    assert jp.allclose(diffusion.ddim_step(noisy, clean, schedule, 0), clean)


def test_step_with_matching_noise_moves_towards_the_clean_sample():
    schedule = diffusion.build_schedule(5)
    clean = jp.full((1, 2, 3), 0.5)
    noisy = clean / schedule.sqrt_recip_alphas[4]
    stepped = diffusion.ddim_step(noisy, clean, schedule, 4)
    expected = clean * jp.sqrt(schedule.alphas_prev[4])
    assert jp.allclose(stepped, expected, atol=1e-6)
