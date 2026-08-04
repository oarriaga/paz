import jax
import jax.numpy as jp
from paz.losses.pix2pose import weighted_reconstruction, WeightedReconstruction


def build_sample(seed):
    key = jax.random.PRNGKey(seed)
    RGB = jax.random.uniform(key, (2, 8, 8, 3))
    alpha = (jax.random.uniform(key, (2, 8, 8, 1)) > 0.5).astype("float32")
    RGBA_true = jp.concatenate([RGB, alpha], axis=-1)
    return RGBA_true, RGB, alpha


def test_perfect_prediction_is_zero():
    RGBA_true, RGB, _ = build_sample(0)
    assert jp.allclose(weighted_reconstruction(RGBA_true, RGB), 0.0)


def test_foreground_is_weighted_by_beta():
    RGBA_true, RGB, alpha = build_sample(1)
    beta = 3.0
    RGB_pred = jp.zeros_like(RGB)
    error = jp.mean(jp.abs(RGB), axis=-1)
    foreground = jp.mean(error * alpha[..., 0])
    background = jp.mean(error * (1.0 - alpha[..., 0]))
    loss = weighted_reconstruction(RGBA_true, RGB_pred, beta)
    expected = beta * error * alpha[..., 0] + error * (1.0 - alpha[..., 0])
    assert jp.allclose(loss, expected)
    assert foreground > 0.0 and background > 0.0


def test_factory_matches_function_and_is_differentiable():
    RGBA_true, RGB, _ = build_sample(2)
    loss_fn = WeightedReconstruction(beta=3.0)
    RGB_pred = jax.random.uniform(jax.random.PRNGKey(3), RGB.shape)
    value = loss_fn(RGBA_true, RGB_pred)
    assert jp.allclose(value, weighted_reconstruction(RGBA_true, RGB_pred, 3.0))
    grad = jax.grad(lambda p: jp.mean(loss_fn(RGBA_true, p)))(RGB_pred)
    assert jp.all(jp.isfinite(grad))
