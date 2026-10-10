import jax.numpy as jp
import keras

import paz


def quadratic_loss(parameters):
    return jp.sum((parameters - 3.0) ** 2)


def test_LBFGS_returns_trimmed_history():
    parameters = jp.array([8.0, -2.0])
    linesearch = paz.optimizers.LineSearch(10, "wolfe")
    fitted, history = paz.optimizers.LBFGS(
        parameters, quadratic_loss, 1.0, 20, 1e-4, 5, linesearch
    )
    assert jp.allclose(fitted, jp.array([3.0, 3.0]), atol=1e-3)
    assert len(history.losses) == int(history.stop_step)


def test_LBFGS_runs_callbacks():
    parameters = jp.array([8.0, -2.0])
    calls = []

    def callback(step_arg, parameters, loss, metrics):
        del parameters, loss, metrics
        calls.append(step_arg)

    linesearch = paz.optimizers.LineSearch(10, "wolfe")
    _, history = paz.optimizers.LBFGS(
        parameters,
        quadratic_loss,
        1.0,
        20,
        1e-4,
        5,
        linesearch,
        callbacks=[callback],
    )
    assert calls[-1] == history.stop_step


def apply_one_step(optimizer, name="w"):
    """Change one gradient step makes, so scaled rates can be compared."""
    variable = keras.Variable(jp.ones(3), name=name)
    optimizer.apply_gradients([(jp.ones(3), variable)])
    return 1.0 - jp.asarray(variable)


def test_LayerwiseAdamW_scales_the_step_of_a_named_variable():
    plain = keras.optimizers.AdamW(0.1, weight_decay=0.0)
    scaled = paz.optimizers.LayerwiseAdamW({"w": 0.5}, 0.1, weight_decay=0.0)
    assert jp.allclose(apply_one_step(scaled), 0.5 * apply_one_step(plain))


def test_LayerwiseAdamW_freezes_a_zero_scale():
    kwargs = dict(weight_decay=0.0)
    optimizer = paz.optimizers.LayerwiseAdamW({"w": 0.0}, 0.1, **kwargs)
    assert jp.allclose(apply_one_step(optimizer), 0.0)


def test_LayerwiseAdamW_leaves_unlisted_variables_alone():
    plain = keras.optimizers.AdamW(0.1, weight_decay=0.0)
    scaled = paz.optimizers.LayerwiseAdamW({"w": 0.5}, 0.1, weight_decay=0.0)
    step = apply_one_step(scaled, "other")
    assert jp.allclose(step, apply_one_step(plain, "other"))
