import jax
import jax.numpy as jp

from paz.inference.metropolis_hastings import sample as mh_sample
from paz.inference.posterior import MCMCPosterior
from paz.inference.latent_space import as_latent_samples, to_inverse_samples
from paz.inference.tuner import AdaptiveStepTuner, Tuner
from paz.inference.utils import validate_space


def infer(key, data, prior, likelihood, method, **kwargs):
    # pgm.infer resolves defaults then overrides; warmup/tune happen here.
    if method is None:
        raise ValueError("method is required")
    method = method.lower()
    if method == "mh":
        return _infer_mh(key, data, prior, likelihood, **kwargs)
    if method in ("smc", "vi", "hmc", "nuts"):
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
    raise ValueError(f"Unknown inference method '{method}'.")


def _infer_mh(key, data, prior, likelihood, **kwargs):
    num_samples = kwargs.get("num_samples")
    if num_samples is None:
        raise ValueError("num_samples is required for MH")
    num_chains = kwargs.get("num_chains", 1)
    sigma = kwargs.get("sigma", None)
    warmup = kwargs.get("warmup", None)
    init = kwargs.get("init", None)
    space = kwargs.get("space", "inv")
    tuner = kwargs.get("tuner", None)
    tune = kwargs.get("tune", None)
    progress = kwargs.get("progress", True)
    validate_space(space)

    # Split once for init/tune/sample to avoid key reuse.
    key_init, key_tune, key_sample = jax.random.split(key, 3)
    positions = _get_initial_positions(key_init, prior, init, num_chains, space)

    def log_density_fn(position):
        log_prior = prior.log_prob(position, space="inv")
        log_likelihood = likelihood.log_prob(position, data, space="inv")
        return log_prior + log_likelihood

    num_warmup = _resolve_num_warmup(num_samples, warmup)
    if sigma is None:
        sigma = tuner.sigma if isinstance(tuner, AdaptiveStepTuner) else 0.1
    run_tuner = _should_tune(tune, tuner)
    if run_tuner:
        sigma = _run_tuner(
            tuner,
            key_tune,
            log_density_fn,
            positions,
            num_chains,
            sigma,
        )

    total_samples = num_samples + num_warmup
    states, infos = mh_sample(
        key_sample,
        log_density_fn,
        positions,
        sigma,
        total_samples,
        num_chains,
        progress=progress,
    )
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
    return MCMCPosterior(states, infos, config, prior.latent_space, "inv")


def _resolve_num_warmup(num_samples, warmup):
    if warmup is None:
        return 0
    if isinstance(warmup, float):
        if warmup < 0 or warmup > 1:
            raise ValueError("warmup fraction must be in [0, 1]")
        return min(int(num_samples * warmup), num_samples)
    if isinstance(warmup, int):
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        return min(warmup, num_samples)
    raise TypeError("warmup must be float or int")


def _get_initial_positions(key, prior, init, num_chains, space):
    if init is None:
        init = prior.sample(key, num_chains, space=space)
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


def _run_tuner(tuner, key, log_density_fn, positions, num_chains, sigma):
    if not isinstance(tuner, AdaptiveStepTuner):
        raise NotImplementedError("Only AdaptiveStepTuner is supported.")
    tune = Tuner(
        log_density_fn,
        positions,
        num_chains,
        compute_rate=tuner.compute_rate,
        progress=tuner.progress,
    )
    tuned_sigma, _ = tune(
        key, tuner.num_steps, tuner.num_episodes, sigma
    )
    return tuned_sigma
