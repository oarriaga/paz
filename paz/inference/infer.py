import inspect

import jax
import jax.numpy as jp

import paz.backend.message as message
import paz.utils.progressbar as progressbar

from paz.inference.metropolis_hastings import sample as mh_sample
from paz.inference.posterior import MCMCPosterior
from paz.inference.latent_space import as_latent_samples, to_inverse_samples
from paz.inference.tuner import Tuner
from paz.inference.utils import validate_space


def infer(key, data, model, method, sample_predictive=None, **kwargs):
    """Samples the posterior of `model` given `data`.

    # Arguments
    - `key`: JAX PRNG key.
    - `data`: Observations passed to `model.log_prob_inverse`.
    - `model`: `PGM`.
    - `method`: String. Only `"mh"` (Metropolis-Hastings) is implemented.
    - `sample_predictive`: `None` or a function `(key, forward_samples)`
      used by the posterior to sample.
    - `kwargs`: Method settings. `"mh"` requires `num_samples` and
      accepts `num_chains`, `sigma`, `warmup`, `init`, `space`, `tuner`,
      `tune`, `progress` and `callbacks`.

    # Returns
    `MCMCPosterior`.
    """
    if method is None:
        raise ValueError("method is required")
    method = method.lower()
    if method == "mh":
        return _infer_mh(key, data, model, sample_predictive, **kwargs)
    if method in ("smc", "vi", "hmc", "nuts"):
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
    raise ValueError(f"Unknown inference method '{method}'.")


def _infer_mh(
    key,
    data,
    model,
    sample_predictive,
    *,
    num_samples=None,
    num_chains=1,
    sigma=None,
    warmup=None,
    init=None,
    space="inv",
    tuner=None,
    tune=None,
    progress=True,
    callbacks=None,
):
    if num_samples is None:
        raise ValueError("num_samples is required for MH")
    validate_space(space)

    # Split once for init/tune/sample to avoid key reuse.
    key_init, key_tune, key_sample = jax.random.split(key, 3)
    init_args = (key_init, model.prior, init, num_chains, space)
    positions = _get_initial_positions(*init_args)

    def log_density_fn(position):
        return model.log_prob_inverse(position, data)

    num_warmup = _resolve_num_warmup(num_samples, warmup)
    if sigma is None:
        sigma = 0.1 if tuner is None else tuner.sigma
    run_tuner = _should_tune(tune, tuner)
    total_samples = num_samples + num_warmup
    callbacks = () if callbacks is None else tuple(callbacks)
    shows_tuning = run_tuner and tuner.progress
    reports = _show_stages(shows_tuning, tuner, progress, total_samples)
    if run_tuner:
        show_tuning = reports.get("tune", progressbar.skip)
        tuner_args = (tuner, key_tune, log_density_fn, positions, sigma)
        sigma, positions = _run_tuner(*tuner_args, show_tuning, callbacks)

    show_sampling = reports.get("sample", progressbar.skip)
    sample_args = (key_sample, log_density_fn, positions, sigma)
    sample_args = sample_args + (total_samples, num_chains, show_sampling)
    states, infos = mh_sample(*sample_args, callbacks)
    if reports:
        progressbar.newline()
    if num_warmup > 0:
        states = jax.tree.map(lambda x: x[num_warmup:], states)
        infos = jax.tree.map(lambda x: x[num_warmup:], infos)

    config = {
        "method": "mh",
        "num_samples": num_samples,
        "num_chains": num_chains,
        "sigma": sigma,
        "warmup": num_warmup,
        "tuner": tuner,
        "tune": run_tuner,
        "space": space,
        "progress": progress,
    }
    posterior_args = (states, infos, config, model.latent_space)
    return MCMCPosterior(*posterior_args, sample_predictive)


def _resolve_num_warmup(num_samples, warmup):
    if warmup is None:
        return 0
    if isinstance(warmup, float):
        if warmup < 0 or warmup > 1:
            raise ValueError("warmup fraction must be in [0, 1]")
        warmup = int(num_samples * warmup)
    elif isinstance(warmup, int):
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
    else:
        raise TypeError("warmup must be float or int")
    return min(warmup, num_samples)


def _get_initial_positions(key, prior, init, num_chains, space):
    if init is None:
        if space == "inv":
            init = prior.sample_inverse(key, num_chains)
        else:
            init = prior.sample(key, num_chains)
    init = as_latent_samples(prior.latent_space, init)
    if space == "fwd":
        init = to_inverse_samples(prior.latent_space, init)

    def repeat_one_sample(value, one_value):
        shape = jp.shape(value)
        if shape == jp.shape(one_value):
            value = jp.broadcast_to(value, (num_chains,) + shape)
        return value

    # A start given as one sample is repeated for every chain.
    one_sample = prior.sample_inverse(key)
    return jax.tree.map(repeat_one_sample, init, one_sample)


def _should_tune(tune, tuner):
    if tune is None:
        return tuner is not None
    if tune and tuner is None:
        raise ValueError("tune=True requires a tuner configuration.")
    return tune


def _show_stages(shows_tuning, tuner, shows_sampling, total_samples):
    stages = []
    if shows_tuning:
        stages.append(("tune", tuner.num_episodes, 10))
    if shows_sampling:
        stages.append(("sample", total_samples, 20))
    names = [description for description, _, _ in stages]
    return dict(zip(names, progressbar.show_stages(stages)))


def _run_tuner(tuner, key, log_density_fn, positions, sigma, show, callbacks):
    tuner_config = {
        "compute_rate": tuner.compute_rate,
        "compute_sigma": tuner.compute_sigma,
        "show_progress": show,
        "callbacks": callbacks,
    }
    tune = Tuner(log_density_fn, positions, **tuner_config)
    tune_args = (key, tuner.num_steps, tuner.num_episodes, sigma)
    result = tune(*tune_args)
    num_invalid = int(jp.sum(~result.is_sigma_valid))
    if num_invalid > 0:
        episodes = f"{num_invalid} of {tuner.num_episodes} episodes"
        message.warn(f"Tuner kept the previous sigma in {episodes}.")
    return result.sigma, result.positions


def find_unknown_settings(method, settings):
    if method != "mh":
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
    parameters = inspect.signature(_infer_mh).parameters.values()
    known = []
    for parameter in parameters:
        if parameter.kind == parameter.KEYWORD_ONLY:
            known.append(parameter.name)
    return sorted(name for name in settings if name not in known)
