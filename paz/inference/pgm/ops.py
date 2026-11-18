import jax
import jax.numpy as jp

from paz.inference.infer import infer as infer_fn
from paz.inference.latent_space import (
    as_latent_samples,
    as_one_sample,
    to_forward_samples,
    to_inverse_samples,
)
from paz.inference.types import NodeState


def build_sample_inverse(graph, prior_sample_inverse):
    def sample_inverse(key, num_samples=1):
        inverse_samples = prior_sample_inverse(key, num_samples)
        return graph.Latent(**inverse_samples._asdict())

    return sample_inverse


def build_sample_forward(graph):
    input_names = [node.name for node in graph.inputs]
    nodes = {node.name: node for node in graph.nodes}
    names = [name for name in graph.Sample._fields if name not in input_names]
    nodes = [nodes[name] for name in names]

    def sample_forward(key, num_samples=1):
        samples = _sample_nodes(graph.inputs, nodes, key, num_samples)
        return graph.Sample(**samples)

    return sample_forward


def build_sample_predictive(graph):
    Sample = graph.Sample
    names = Sample._fields
    nodes = graph.nodes
    latent_space = graph.latent_space
    latent_names = set(latent_space.names)
    name_to_node = {node.name: node for node in nodes}
    nodes_sorted = [name_to_node[name] for name in names]

    def predictive_single(subkey, sample):
        return _sample_predictive_single(
            subkey, sample, nodes_sorted, latent_names, Sample
        )

    def sample_predictive(key, forward_samples):
        forward_samples = as_latent_samples(latent_space, forward_samples)
        num_samples = jax.tree.leaves(forward_samples)[0].shape[0]
        keys = jax.random.split(key, num_samples)
        return jax.vmap(predictive_single)(keys, forward_samples)

    return sample_predictive


# Latents are drawn in forward space, where their parents are valid
# distribution parameters, and are moved to inverse space afterwards.
def build_prior_sample(graph):
    latent_space = graph.latent_space
    input_names = [node.name for node in graph.inputs]
    nodes = graph.latent_nodes_sorted
    nodes = [node for node in nodes if node.name not in input_names]

    def prior_sample(key, num_samples=1):
        samples = _sample_nodes(graph.inputs, nodes, key, num_samples)
        return as_latent_samples(latent_space, samples)

    def prior_sample_inverse(key, num_samples=1):
        forward_samples = prior_sample(key, num_samples)
        return to_inverse_samples(latent_space, forward_samples)

    return prior_sample, prior_sample_inverse


# One draw with a fixed key gives the shape of one sample of every latent.
def compute_sample_shapes(sample):
    return jax.tree.map(jp.shape, sample(jax.random.PRNGKey(0)))


def build_prior_log_prob(graph, shapes, inverse_shapes):
    nodes = graph.latent_nodes_sorted
    latent_space = graph.latent_space

    def prior_log_prob(samples):
        forward_samples = as_one_sample(latent_space, samples, shapes)
        log_probs = compute_log_probs(nodes, forward_samples._asdict())
        return build_node_state(forward_samples, log_probs)

    def prior_log_prob_inverse(samples):
        args = (latent_space, samples, inverse_shapes)
        inverse_samples = as_one_sample(*args)
        forward_samples = to_forward_samples(latent_space, inverse_samples)
        values = forward_samples._asdict()
        inverse_values = inverse_samples._asdict()
        log_probs = compute_inverse_log_probs(nodes, values, inverse_values)
        return build_node_state(forward_samples, log_probs)

    return prior_log_prob, prior_log_prob_inverse


def compute_log_probs(nodes, values):
    log_probs = {}
    for node in nodes:
        parents = [values[edge.name] for edge in node.edges]
        state = node.log_prob(values[node.name], *parents)
        log_probs[node.name] = state.log_prob_sum
    return log_probs


def build_node_state(sample, log_probs):
    log_prob_sum = sum(log_probs.values(), jp.array(0.0))
    return NodeState(sample, log_probs, log_prob_sum)


def compute_inverse_log_probs(nodes, values, inverse_values):
    log_probs = {}
    for node in nodes:
        parents = [values[edge.name] for edge in node.edges]
        state = node.log_prob_inverse(inverse_values[node.name], *parents)
        log_probs[node.name] = state.log_prob_sum
    return log_probs


def build_prior_prob(prior_log_prob, prior_log_prob_inverse):
    def prior_prob(samples):
        return jp.exp(prior_log_prob(samples).log_prob_sum)

    def prior_prob_inverse(samples):
        return jp.exp(prior_log_prob_inverse(samples).log_prob_sum)

    return prior_prob, prior_prob_inverse


def build_likelihood_log_prob(graph, shapes, inverse_shapes):
    latent_space, Sample = graph.latent_space, graph.Sample
    nodes, output_nodes = graph.observable_nodes, graph.output_nodes

    def compute_state(forward_samples, observations):
        values = {**forward_samples._asdict(), **observations}
        log_probs = compute_log_probs(nodes, values)
        return build_node_state(Sample(**values), log_probs)

    def likelihood_log_prob(samples, data):
        observations = build_observations(data, output_nodes, nodes)
        forward_samples = as_one_sample(latent_space, samples, shapes)
        return compute_state(forward_samples, observations)

    def likelihood_log_prob_inverse(samples, data):
        observations = build_observations(data, output_nodes, nodes)
        args = (latent_space, samples, inverse_shapes)
        forward_samples = to_forward_samples(latent_space, as_one_sample(*args))
        return compute_state(forward_samples, observations)

    return likelihood_log_prob, likelihood_log_prob_inverse


def build_observations(data, output_nodes, observable_nodes):
    names = [node.name for node in observable_nodes]
    data_mapping = build_data_mapping(data, output_nodes)
    _validate_observations(data_mapping, names)
    return {name: data_mapping[name] for name in names}


def build_log_density(prior, likelihood):
    def log_prob(samples, data):
        log_prior = prior.log_prob(samples).log_prob_sum
        log_likelihood = likelihood.log_prob(samples, data).log_prob_sum
        return log_prior + log_likelihood

    def log_prob_inverse(samples, data):
        log_prior = prior.log_prob_inverse(samples).log_prob_sum
        log_likelihood = likelihood.log_prob_inverse(samples, data)
        return log_prior + log_likelihood.log_prob_sum

    return log_prob, log_prob_inverse


def build_configure(inference_defaults):
    def configure(
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

    return configure


def build_infer(get_model, inference_defaults, sample_predictive=None):
    def infer(key, data, method=None, tune=None, **overrides):
        method_name = method or inference_defaults.get("_compiled_method", "mh")
        method_name = method_name.lower()
        resolved = dict(inference_defaults.get(method_name, {}))
        if tune is not None:
            overrides["tune"] = tune
        resolved.update(overrides)
        resolved["sample_predictive"] = sample_predictive
        return infer_fn(key, data, get_model(), method_name, **resolved)

    return infer


def build_fit(fitters=None):
    fitters = {} if fitters is None else dict(fitters)

    def fit(key, data, method=None, **method_kwargs):
        if not fitters:
            raise NotImplementedError("fit is not implemented for this model.")
        method_name = method.lower() if method is not None else None
        if method_name is None:
            raise ValueError("fit method is required.")
        fit_fn = fitters.get(method_name)
        if fit_fn is None:
            raise ValueError("Unknown fit method: " + method_name)
        return fit_fn(key, data, **method_kwargs)

    return fit


def build_data_mapping(data, output_nodes):
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "_asdict"):
        return data._asdict()
    if isinstance(data, (list, tuple)):
        if len(data) != len(output_nodes):
            raise ValueError("Data list must match the number of PGM outputs.")
        return {node.name: value for node, value in zip(output_nodes, data)}
    if len(output_nodes) == 1:
        return {output_nodes[0].name: data}
    message = (
        "Data must be a dict, list aligned to outputs, or a single "
        "observation for single-output models."
    )
    raise TypeError(message)


def _validate_observations(data_mapping, observable_names):
    missing = [name for name in observable_names if name not in data_mapping]
    missing.sort()
    if missing:
        raise ValueError("Missing observations for: " + ", ".join(missing))


def _sample_nodes(prior_nodes, non_prior_nodes, key, num_samples):
    key_prior, key_node = jax.random.split(key)
    samples, keys = {}, jax.random.split(key_prior, len(prior_nodes))
    for subkey, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample(subkey, num_samples)
    keys = jax.random.split(key_node, len(non_prior_nodes))
    for subkey, node in zip(keys, non_prior_nodes):
        parents = [samples[edge.name] for edge in node.edges]
        args = (node, subkey, num_samples, parents)
        samples[node.name] = _sample_with_parents(*args)
    return samples


def _sample_with_parents(node, key, num_samples, parents):
    if num_samples == 1 or len(parents) == 0:
        sample = node.sample(key, num_samples, *parents)
    else:
        subkeys = jax.random.split(key, num_samples)

        def sample_once(subkey, *parent_values):
            return node.sample(subkey, 1, *parent_values)

        sample = jax.vmap(sample_once)(subkeys, *parents)
    return sample


def _sample_predictive_single(
    key, forward_samples, nodes_sorted, latent_names, Sample
):
    samples = {}
    keys = jax.random.split(key, len(nodes_sorted))
    for node, subkey in zip(nodes_sorted, keys):
        if node.name in latent_names:
            samples[node.name] = getattr(forward_samples, node.name)
            continue
        node_inputs = [samples[edge.name] for edge in node.edges]
        if node.sample is None:
            raise ValueError("Node sample is missing for " + node.name)
        samples[node.name] = node.sample(subkey, 1, *node_inputs)
    return Sample(**samples)
