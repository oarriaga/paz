import jax
import jax.numpy as jp

from paz.inference.infer import infer as infer_fn
from paz.inference.metadata import get_distribution_fn
from paz.inference.latent_space import (
    as_latent_samples,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.types import Distribution, NodeState
from paz.inference.utils import validate_space


def build_sample_inverse(context):
    return lambda key, num_samples=1: context.Latent(
        **_sample_inverse(
            context.inputs, context.latent_nodes, key, num_samples
        )
    )


def build_sample_forward(context):
    return lambda key, num_samples=1: context.Sample(
        **_sample_forward(
            context.inputs, context.non_priors, key, num_samples
        )
    )


def build_apply(context):
    Sample = context.Sample
    inputs = context.inputs
    non_priors = context.non_priors
    output_nodes = context.output_nodes
    observable_names = context.observable_names

    def apply(inverse_samples, data=None):
        data_mapping = build_data_mapping(data, output_nodes)
        _validate_observations(data_mapping, observable_names)
        sample, log_prob = _apply(
            inputs, non_priors, inverse_samples, data_mapping, observable_names
        )
        log_prob_sum = sum(log_prob.values(), jp.array(0.0))
        return NodeState(Sample(**sample), log_prob, log_prob_sum)

    return apply


def build_prior_sample(context, sample_inverse):
    latent_space = context.latent_space

    def prior_sample(key, num_samples=1, space="inv"):
        validate_space(space)
        inverse_samples = sample_inverse(key, num_samples)
        inverse_samples = as_latent_samples(latent_space, inverse_samples)
        if space == "inv":
            return inverse_samples
        return to_forward_samples(latent_space, inverse_samples)

    return prior_sample


def build_prior_log_prob(context):
    latent_nodes_sorted = context.latent_nodes_sorted
    latent_space = context.latent_space

    def prior_log_prob(samples, space="inv"):
        validate_space(space)
        if space == "inv":
            inverse_samples = as_latent_samples(latent_space, samples)
            forward_samples = to_forward_samples(latent_space, inverse_samples)
        else:
            inverse_samples = None
            forward_samples = as_latent_samples(latent_space, samples)
        log_prob = _compute_latent_log_probs(
            latent_nodes_sorted,
            forward_samples,
            inverse_samples,
            latent_space.bijectors,
            include_jacobian=space == "inv",
        )
        return sum(log_prob.values(), jp.array(0.0))

    return prior_log_prob


def build_prior_prob(prior_log_prob):
    return lambda samples, space="inv": jp.exp(
        prior_log_prob(samples, space=space)
    )


def build_likelihood_log_prob(context, apply):
    output_nodes = context.output_nodes
    observable_names = context.observable_names
    latent_space = context.latent_space

    def likelihood_log_prob(samples, data, space="inv"):
        validate_space(space)
        data_mapping = build_data_mapping(data, output_nodes)
        _validate_observations(data_mapping, observable_names)
        if space == "fwd":
            inverse_samples = to_inverse_samples(latent_space, samples)
        else:
            inverse_samples = as_latent_samples(latent_space, samples)
        state = apply(inverse_samples, data_mapping)
        return sum(
            (state.log_prob[name] for name in observable_names),
            jp.array(0.0),
        )

    return likelihood_log_prob


def build_compile(inference_defaults, get_self):
    # compile stores per-method defaults; infer merges defaults then overrides.
    def compile(
        method="mh",
        num_chains=None,
        warmup=None,
        tuner=None,
        **method_kwargs,
    ):
        method = method.lower()
        defaults = inference_defaults.setdefault(method, {})
        if num_chains is not None:
            defaults["num_chains"] = num_chains
        if warmup is not None:
            defaults["warmup"] = warmup
        if tuner is not None:
            defaults["tuner"] = tuner
        defaults.update(method_kwargs)
        inference_defaults["_compiled_method"] = method
        return get_self()

    return compile


def build_tune(compile_fn):
    def tune(method="mh", **kwargs):
        return compile_fn(method=method, **kwargs)

    return tune


def build_infer(pgm_prior, pgm_likelihood, inference_defaults):
    def infer(
        key,
        data,
        prior=None,
        likelihood=None,
        method=None,
        tune=None,
        **overrides,
    ):
        method_name = method or inference_defaults.get("_compiled_method", "mh")
        method_name = method_name.lower()
        resolved = dict(inference_defaults.get(method_name, {}))
        if tune is not None:
            overrides["tune"] = tune
        resolved.update(overrides)
        prior = prior if prior is not None else pgm_prior
        likelihood = likelihood if likelihood is not None else pgm_likelihood
        return infer_fn(key, data, prior, likelihood, method_name, **resolved)

    return infer


def build_data_mapping(data, output_nodes):
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "_asdict"):
        return data._asdict()
    if isinstance(data, (list, tuple)):
        if len(data) != len(output_nodes):
            raise ValueError(
                "Data list must match the number of PGM outputs."
            )
        return {node.name: value for node, value in zip(output_nodes, data)}
    if len(output_nodes) == 1:
        return {output_nodes[0].name: data}
    raise TypeError(
        "Data must be a dict, list aligned to outputs, or a single "
        "observation for single-output models."
    )


def get_namedtuple_value(field, named_tuple, default=None):
    if isinstance(named_tuple, Distribution):
        return named_tuple
    return getattr(named_tuple, field, default)


def _validate_observations(data_mapping, observable_names):
    missing = sorted(
        name for name in observable_names if name not in data_mapping
    )
    if missing:
        raise ValueError(
            "Missing observations for: " + ", ".join(missing)
        )


def _sample_priors(prior_nodes, key, num_samples, sample_fn):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = sample_fn(prior, key, num_samples)
    return samples


def _sample_with_inputs(node, sample_fn, key, num_samples, node_inputs):
    if num_samples == 1 or len(node_inputs) == 0:
        return sample_fn(node, key, num_samples, *node_inputs)
    subkeys = jax.random.split(key, num_samples)

    def sample_node(subkey, *inputs):
        return sample_fn(node, subkey, 1, *inputs)

    in_axes = (0,) + (0,) * len(node_inputs)
    return jax.vmap(sample_node, in_axes=in_axes)(subkeys, *node_inputs)


def _sample_nodes(
    prior_nodes,
    non_prior_nodes,
    key,
    num_samples,
    prior_sample_fn,
    node_sample_fn,
):
    key_prior, key_node = jax.random.split(key)
    samples = _sample_priors(
        prior_nodes, key_prior, num_samples, prior_sample_fn
    )
    keys = jax.random.split(key_node, len(non_prior_nodes))
    for key, node in zip(keys, non_prior_nodes):
        node_inputs = [samples[edge.name] for edge in node.edges]
        node_sample = _sample_with_inputs(
            node, node_sample_fn, key, num_samples, node_inputs
        )
        samples[node.name] = node_sample
    return samples


def _sample_inverse(prior_nodes, latent_nodes, key, num_samples):
    prior_names = {prior.name for prior in prior_nodes}
    non_prior_latents = [n for n in latent_nodes if n.name not in prior_names]
    return _sample_nodes(
        prior_nodes,
        non_prior_latents,
        key,
        num_samples,
        lambda prior, subkey, num: prior.sample_inverse(subkey, num),
        lambda node, subkey, num, *inputs: node.sample_inverse(
            subkey, num, *inputs
        ),
    )


def _sample_forward(prior_nodes, non_prior_nodes, key, num_samples):
    return _sample_nodes(
        prior_nodes,
        non_prior_nodes,
        key,
        num_samples,
        lambda prior, subkey, num: prior.sample(subkey, num),
        lambda node, subkey, num, *inputs: node.sample(subkey, num, *inputs),
    )


def _apply(priors, non_priors, inverse_samples, data_mapping, observable_names):
    samples, log_prob = {}, {}
    for prior in priors:
        state = prior.apply(get_namedtuple_value(prior.name, inverse_samples))
        samples[prior.name] = get_namedtuple_value(prior.name, state.sample)
        log_prob[prior.name] = state.log_prob.sum()
    for node in non_priors:
        node_inputs = [samples[edge.name] for edge in node.edges]
        if node.name in observable_names:
            observation = data_mapping[node.name]
            state = node.apply(observation, *node_inputs)
            samples[node.name] = observation
        else:
            node_sample = get_namedtuple_value(node.name, inverse_samples)
            state = node.apply(node_sample, *node_inputs)
            samples[node.name] = get_namedtuple_value(node.name, state.sample)
        log_prob[node.name] = state.log_prob.sum()
    return samples, log_prob


def _compute_latent_log_probs(
    latent_nodes, forward_samples, inverse_samples, bijectors, include_jacobian
):
    log_prob = {}
    for node in latent_nodes:
        forward_sample = getattr(forward_samples, node.name)
        if node.distribution is not None:
            distribution = node.distribution
        else:
            distribution_fn = get_distribution_fn(node)
            if distribution_fn is None:
                raise ValueError("Latent node missing distribution_fn.")
            parent_samples = [
                getattr(forward_samples, edge.name) for edge in node.edges
            ]
            distribution = distribution_fn(*parent_samples)
        log_prob_value = distribution.log_prob(forward_sample)
        if include_jacobian:
            if inverse_samples is None:
                raise ValueError("Inverse samples required for Jacobian.")
            bijector = bijectors[node.name]
            inverse_sample = getattr(inverse_samples, node.name)
            log_prob_value = (
                log_prob_value
                + bijector.forward_log_det_jacobian(inverse_sample)
            )
        log_prob[node.name] = log_prob_value.sum()
    return log_prob
