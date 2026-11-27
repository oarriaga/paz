import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior


def build_simple_model():
    x = Prior(paz.distributions.Normal(0.0, 1.0), name="x")

    def likelihood(x_value):
        return paz.distributions.Normal(x_value, 1.0)

    y = Observable(likelihood, name="y")(x)
    data = {"y": jp.array(0.25)}
    return PGM([x], [y], "simple"), data


def test_infer_mh_shapes():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(0)
    posterior = model.infer(
        key,
        data,
        method="mh",
        num_samples=10,
        num_chains=2,
        sigma=0.2,
        progress=False,
    )
    positions = posterior.inverse_samples
    assert positions.x.shape == (10, 2)
    assert posterior.inverse_log_probs.shape == (10, 2)


def test_infer_mh_uses_tuned_defaults():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(1)
    infer_key = jax.random.split(key)[0]
    model.configure(
        num_chains=3,
        sigma=0.3,
        warmup=0,
        tuner=paz.AdaptiveStepTuner(
            sigma=0.3, num_steps=2, num_episodes=1, progress=False
        ),
    )
    posterior = model.infer(infer_key, data, num_samples=5, tune=False)
    assert bool(jp.isclose(posterior.config["sigma"], 0.3))
    assert posterior.config["num_chains"] == 3
    assert posterior.config["num_samples"] == 5


def test_infer_mh_warmup_discards_samples():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(2)
    posterior = model.infer(
        key,
        data,
        method="mh",
        num_samples=5,
        num_chains=1,
        warmup=2,
        progress=False,
    )
    positions = posterior.inverse_samples
    assert positions.x.shape == (5, 1)
    assert posterior.config["warmup"] == 2


def test_infer_mh_unknown_method_raises():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(3)
    with pytest.raises(NotImplementedError):
        model.infer(
            key,
            data,
            method="smc",
            num_samples=5,
        )


def test_tune_updates_default_sigma():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(4)
    infer_key = jax.random.split(key)[0]
    model.configure(
        num_chains=1,
        sigma=0.4,
        warmup=0,
        tuner=paz.AdaptiveStepTuner(
            sigma=0.4, num_steps=2, num_episodes=1, progress=False
        ),
    )
    posterior = model.infer(
        infer_key,
        data,
        num_samples=5,
        num_chains=1,
        progress=False,
    )
    assert posterior.config["tune"] is True


diagonal_case = (paz.AdaptiveDiagonalTuner, (1,))
covariance_case = (paz.AdaptiveCovarianceTuner, (1, 1))


@pytest.mark.parametrize("Tuner, shape", [diagonal_case, covariance_case])
def test_infer_mh_with_shape_tuners(Tuner, shape):
    model, data = build_simple_model()
    tuner = Tuner(sigma=0.4, num_steps=50, num_episodes=4, progress=False)
    model.configure(num_chains=4, warmup=0, tuner=tuner, progress=False)
    posterior = model.infer(jax.random.PRNGKey(5), data, num_samples=2000)
    samples = posterior.samples.x
    assert posterior.config["sigma"].shape == shape
    assert jp.abs(samples.mean() - 0.125) < 0.1
    assert jp.abs(samples.std() - jp.sqrt(0.5)) < 0.1


def infer_with_progress(tuner_progress, capsys):
    model, data = build_simple_model()
    kwargs = {"num_steps": 5, "num_episodes": 2, "progress": tuner_progress}
    tuner = paz.AdaptiveStepTuner(sigma=0.4, **kwargs)
    model.configure(num_chains=2, warmup=0, tuner=tuner, progress=True)
    model.infer(jax.random.PRNGKey(6), data, num_samples=5)
    jax.effects_barrier()
    return capsys.readouterr().out


def test_infer_mh_shows_tuning_and_sampling_on_one_line(capsys):
    output = infer_with_progress(True, capsys)
    last_frame = output.rstrip().split("\r")[-1]
    assert output.count("\n") == 1
    assert f"tune |{'█' * 10}| 2/2" in last_frame
    assert f"sample |{'█' * 20}| 5/5" in last_frame


def test_infer_mh_shows_only_sampling_without_tuner_progress(capsys):
    output = infer_with_progress(False, capsys)
    assert output.count("\n") == 1
    assert "sample |" in output
    assert "tune" not in output


def test_infer_rejects_unknown_setting():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(0)
    with pytest.raises(TypeError, match="num_chain"):
        model.infer(key, data, num_samples=5, num_chain=2)


def test_infer_repeats_one_vector_start_for_every_chain():
    normal = paz.distributions.Normal(jp.zeros(3), 1.0)
    a = Prior(normal, name="a")
    y = Observable(lambda a: paz.distributions.Normal(a, 1.0), name="y")(a)
    model = PGM([a], [y], "vector_start")
    kwargs = {"num_samples": 4, "num_chains": 3, "sigma": 0.1}
    kwargs = {**kwargs, "init": {"a": jp.zeros(3)}, "progress": False}
    posterior = model.infer(jax.random.PRNGKey(0), {"y": jp.zeros(3)}, **kwargs)
    assert posterior.inverse_samples.a.shape == (4, 3, 3)


def test_configure_rejects_unknown_setting():
    model, data = build_simple_model()
    with pytest.raises(TypeError, match="num_chain"):
        model.configure(num_chain=2)
    model.configure(num_chains=2, sigma=0.2, progress=False)
    posterior = model.infer(jax.random.PRNGKey(0), data, num_samples=5)
    assert posterior.inverse_samples.x.shape == (5, 2)


def test_configure_rejects_unknown_method():
    model, data = build_simple_model()
    with pytest.raises(NotImplementedError):
        model.configure(method="mhh", sigma=0.2)


def test_infer_mh_callbacks_receive_every_tuning_and_sampling_step():
    model, data = build_simple_model()
    calls = []

    def callback(stage, step_arg, samples, acceptance_rate):
        shapes = (samples.position.x.shape, samples.log_density.shape)
        calls.append((stage, int(step_arg), shapes, acceptance_rate.shape))

    tuner = paz.AdaptiveCovarianceTuner(0.2, 5, 3, progress=False)
    kwargs = dict(num_samples=6, num_chains=2, warmup=2, tuner=tuner)
    kwargs.update(progress=False, callbacks=[callback])
    model.infer(jax.random.PRNGKey(0), data, method="mh", **kwargs)
    shapes = (((2,), (2,)), (2,))
    expected = [("tune", episode, *shapes) for episode in range(3)]
    expected += [("sample", step, *shapes) for step in range(8)]
    assert calls == expected


def test_infer_mh_callbacks_receive_the_samples_of_the_chain():
    model, data = build_simple_model()
    positions = []

    def callback(stage, step_arg, samples, acceptance_rate):
        positions.append(samples.position.x)

    kwargs = dict(method="mh", num_samples=6, num_chains=2, sigma=0.2)
    kwargs.update(warmup=2, progress=False, callbacks=[callback])
    posterior = model.infer(jax.random.PRNGKey(0), data, **kwargs)
    assert jp.array_equal(jp.stack(positions[2:]), posterior.inverse_samples.x)


def test_infer_mh_callbacks_do_not_change_the_samples():
    model, data = build_simple_model()
    kwargs = dict(method="mh", num_samples=10, num_chains=2, sigma=0.2)
    kwargs.update(progress=False)
    key = jax.random.PRNGKey(0)
    plain = model.infer(key, data, **kwargs)
    observed = model.infer(key, data, callbacks=[lambda *args: None], **kwargs)
    assert jp.array_equal(plain.inverse_samples.x, observed.inverse_samples.x)
