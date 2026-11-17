import time

import jax
import jax.numpy as jp

import paz.backend.message as message
import paz.utils.progressbar as progressbar

from paz.inference.metropolis_hastings import sample as mh_sample
from paz.inference.posterior import MCMCPosterior
from paz.inference.latent_space import as_latent_samples, to_inverse_samples
from paz.inference.tuner import Tuner
from paz.inference.utils import validate_space


def infer(
    key, data, prior, likelihood, method, sample_predictive=None, **kwargs
):
    if method is None:
        raise ValueError("method is required")
    method = method.lower()
    if method == "mh":
        return _infer_mh(
            key, data, prior, likelihood, sample_predictive, **kwargs
        )
    if method in ("smc", "vi", "hmc", "nuts"):
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
    raise ValueError(f"Unknown inference method '{method}'.")


def build_log_density(prior, likelihood):
    """Builds the joint log density: log prior plus log likelihood.

    # Arguments
        prior: Density of the PGM priors.
        likelihood: Likelihood of the PGM observables.

    # Returns
        log_prob(samples, data) and log_prob_inverse(samples, data), each
        returning the scalar joint log density. The per-node terms remain
        available from prior.log_prob* and likelihood.log_prob*.
    """

    def log_prob(samples, data):
        log_prior = prior.log_prob(samples).log_prob_sum
        log_likelihood = likelihood.log_prob(samples, data).log_prob_sum
        return log_prior + log_likelihood

    def log_prob_inverse(samples, data):
        log_prior = prior.log_prob_inverse(samples).log_prob_sum
        log_likelihood = likelihood.log_prob_inverse(samples, data)
        return log_prior + log_likelihood.log_prob_sum

    return log_prob, log_prob_inverse


def _infer_mh(key, data, prior, likelihood, sample_predictive, **kwargs):
    num_samples = kwargs.get("num_samples")
    if num_samples is None:
        raise ValueError("num_samples is required for MH")
    settings = _resolve_mh_settings(kwargs)
    num_chains = settings["num_chains"]
    sigma = settings["sigma"]
    warmup = settings["warmup"]
    init = settings["init"]
    space = settings["space"]
    tuner = settings["tuner"]
    tune = settings["tune"]
    progress = settings["progress"]
    validate_space(space)

    # Split once for init/tune/sample to avoid key reuse.
    key_init, key_tune, key_sample = jax.random.split(key, 3)
    positions = _get_initial_positions(key_init, prior, init, num_chains, space)

    _, log_prob_inverse = build_log_density(prior, likelihood)

    def log_density_fn(position):
        return log_prob_inverse(position, data)

    num_warmup = _resolve_num_warmup(num_samples, warmup)
    if sigma is None:
        sigma = 0.1 if tuner is None else tuner.sigma
    run_tuner = _should_tune(tune, tuner)
    total_samples = num_samples + num_warmup
    shares_line = run_tuner and tuner.progress and progress
    show_sample = None
    if run_tuner:
        show_tune = _show_tuning(shares_line, tuner, total_samples)
        tuner_args = (tuner, key_tune, log_density_fn, positions, sigma)
        start_time = time.perf_counter()
        sigma, positions = _run_tuner(*tuner_args, show_tune)
        seconds = time.perf_counter() - start_time
        show_args = (shares_line, tuner, seconds, total_samples)
        show_sample = _show_sampling(*show_args)

    sample_args = (key_sample, log_density_fn, positions, sigma)
    sample_kwargs = {"progress": progress, "show_progress": show_sample}
    sample_args = sample_args + (total_samples, num_chains)
    states, infos = mh_sample(*sample_args, **sample_kwargs)
    if shares_line:
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
    return MCMCPosterior(
        states, infos, config, prior.latent_space, sample_predictive
    )


def _resolve_mh_settings(kwargs):
    defaults = {
        "num_chains": 1,
        "sigma": None,
        "warmup": None,
        "init": None,
        "space": "inv",
        "tuner": None,
        "tune": None,
        "progress": True,
    }
    return {
        name: kwargs.get(name, default) for name, default in defaults.items()
    }


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

    def ensure(value):
        if not hasattr(value, "shape"):
            return value
        if value.shape == ():
            return jp.broadcast_to(value, (num_chains,))
        if value.shape[0] == num_chains:
            return value
        return jp.broadcast_to(value, (num_chains,) + value.shape)

    return jax.tree.map(ensure, init)


def _should_tune(tune, tuner):
    if tune is None:
        return tuner is not None
    if tune and tuner is None:
        raise ValueError("tune=True requires a tuner configuration.")
    return tune


def _show_tuning(shares_line, tuner, total_samples):
    if shares_line:
        waiting = progressbar.build_stage("sample", 0, total_samples, 20)

        def build_text(episode):
            stage_args = ("tune", episode, tuner.num_episodes, 10)
            return f"{progressbar.build_stage(*stage_args)}  {waiting}"

        show = progressbar.show_text(build_text)
    else:
        show = None
    return show


def _show_sampling(shares_line, tuner, seconds, total_samples):
    if shares_line:
        stage_args = ("tune", tuner.num_episodes, tuner.num_episodes, 10)
        tuned = f"{progressbar.build_stage(*stage_args)} {seconds:.2f}s"
        start_time = progressbar.start()

        def build_text(step):
            stage = progressbar.build_stage("sample", step, total_samples, 20)
            timing = progressbar.build_timing(step, total_samples, start_time)
            return f"{tuned}  {stage} {timing}"

        show = progressbar.show_text(build_text)
    else:
        show = None
    return show


def _run_tuner(tuner, key, log_density_fn, positions, sigma, show_progress):
    tuner_config = {
        "compute_rate": tuner.compute_rate,
        "progress": tuner.progress,
        "compute_sigma": tuner.compute_sigma,
        "show_progress": show_progress,
    }
    tune = Tuner(log_density_fn, positions, **tuner_config)
    tune_args = (key, tuner.num_steps, tuner.num_episodes, sigma)
    result = tune(*tune_args)
    num_invalid = int(jp.sum(~result.is_sigma_valid))
    if num_invalid > 0:
        episodes = f"{num_invalid} of {tuner.num_episodes} episodes"
        message.warn(f"Tuner kept the previous sigma in {episodes}.")
    return result.sigma, result.positions
