import jax.numpy as jp
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
