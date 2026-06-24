import jax
import jax.numpy as jp
import paz


def build_masks(seed):
    key = jax.random.PRNGKey(seed)
    logits = jax.random.normal(key, (2, 16, 16, 4))
    y_pred = jax.nn.softmax(logits, axis=-1)
    labels = jax.random.randint(key, (2, 16, 16), 0, 4)
    y_true = jax.nn.one_hot(labels, 4)
    return y_true, y_pred


def test_dice_perfect_prediction_is_zero():
    y_true, _ = build_masks(0)
    assert jp.allclose(paz.losses.dice(y_true, y_true), 0.0, atol=1e-3)


def test_jaccard_perfect_prediction_is_zero():
    y_true, _ = build_masks(1)
    assert jp.allclose(paz.losses.jaccard(y_true, y_true), 0.0, atol=1e-3)


def test_segmentation_losses_are_finite_and_differentiable():
    y_true, y_pred = build_masks(2)
    for loss in [paz.losses.dice, paz.losses.jaccard, paz.losses.focal]:
        value = loss(y_true, y_pred)
        assert jp.all(jp.isfinite(value))
        grad = jax.grad(lambda p: jp.mean(loss(y_true, p)))(y_pred)
        assert jp.all(jp.isfinite(grad))
