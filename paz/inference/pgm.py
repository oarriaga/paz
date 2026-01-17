import jax

from paz.inference.types import (
    Distribution,
    NodeState,
    PGMMetadata,
    SampleType,
    Variable,
)
from paz.abstract.dag import DAG


def search_nodes(output_nodes):
    tree_nodes, queue = [], list(output_nodes)
    while len(queue) != 0:
        node = queue.pop(0)
        queue.extend(node.edges)
        if node not in tree_nodes:
            tree_nodes.append(node)
    return tree_nodes


def _validate_unique_names(nodes):
    name_counts = {}
    for node in nodes:
        name_counts[node.name] = name_counts.get(node.name, 0) + 1
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        duplicates.sort()
        raise ValueError(
            "PGM node names must be unique. "
            f"Duplicates: {', '.join(duplicates)}"
        )


def get_edges(nodes):
    edges = []
    for node in nodes:
        for edge in node.edges:
            source = edge.name
            target = node.name
            edges.append([source, target])
    return edges


def get_non_root_nodes(nodes, sorted_names, prior_names):
    non_prior_names = set(sorted_names) - set(prior_names)
    name_to_node = {node.name: node for node in nodes}
    return [
        name_to_node[name] for name in sorted_names if name in non_prior_names
    ]


def _sample_inverse_priors(prior_nodes, key, num_samples):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample_inverse(key, num_samples)
    return samples


def get_edge_names(node):
    return [edge.name for edge in node.edges]


def get_values_by_key(keys, dictionary):
    return [dictionary[key] for key in keys]


def get_node_inputs(node, dictionary):
    return get_values_by_key(get_edge_names(node), dictionary)


def sample_inverse_latent(node, key, num_samples, node_inputs):
    # Samples inverse values for Latent nodes with parent dependencies.
    # When num_samples > 1, node_inputs are batched with leading dimension.
    # Uses vmap to sample per chain, avoiding broadcasting batched parents into
    # scalar distributions (which would create incorrect extra dimensions).
    if num_samples == 1:
        return node.sample_inverse(key, 1, *node_inputs)
    subkeys = jax.random.split(key, num_samples)

    def sample_latent(subkey, *inputs):
        return node.sample_inverse(subkey, 1, *inputs)

    in_axes = (0,) + (0,) * len(node_inputs)
    return jax.vmap(sample_latent, in_axes=in_axes)(subkeys, *node_inputs)


def _sample_inverse(prior_nodes, latent_nodes, key, num_samples):
    # Samples latent variables (for MCMC initialization).
    # latent_nodes includes ALL nodes with sample_inverse: Priors + Latents.
    # Must filter to avoid double-sampling priors (already sampled in first step).
    # Returns: {node_name: inverse_sample} for all latent variables.
    key_prior, key_node = jax.random.split(key)
    samples = _sample_inverse_priors(prior_nodes, key_prior, num_samples)
    prior_names = {prior.name for prior in prior_nodes}
    non_prior_latents = [n for n in latent_nodes if n.name not in prior_names]
    keys = jax.random.split(key_node, len(non_prior_latents))
    for key, node in zip(keys, non_prior_latents):
        node_inputs = get_node_inputs(node, samples)
        # Latents depend on parent values, so sample per chain when batched.
        node_inverse_sample = sample_inverse_latent(
            node, key, num_samples, node_inputs
        )
        samples[node.name] = node_inverse_sample
    return samples


def _sample_forward_priors(prior_nodes, key, num_samples):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample(key, num_samples)
    return samples


def sample_forward_node(node, key, num_samples, node_inputs):
    # Samples forward values for nodes with parent dependencies (Latent/Observable).
    # When num_samples > 1 and node has parents, node_inputs are batched.
    # Uses vmap to sample per chain, avoiding broadcasting batched parents into
    # batched distributions (which would create shape (num_samples, num_samples, ...)).
    if num_samples == 1 or len(node_inputs) == 0:
        return node.sample(key, num_samples, *node_inputs)
    subkeys = jax.random.split(key, num_samples)

    def sample_node(subkey, *inputs):
        return node.sample(subkey, 1, *inputs)

    in_axes = (0,) + (0,) * len(node_inputs)
    return jax.vmap(sample_node, in_axes=in_axes)(subkeys, *node_inputs)


def _sample_forward(prior_nodes, non_prior_nodes, key, num_samples):
    # Samples all variables (for prior/posterior predictive sampling).
    # non_prior_nodes excludes Priors (pre-filtered by get_non_root_nodes).
    # Includes: Latents + Observables.
    # No filtering needed - priors already excluded from input.
    # Returns: {node_name: forward_sample} for all nodes (priors + latents + observables).
    key_prior, key_node = jax.random.split(key)
    samples = _sample_forward_priors(prior_nodes, key_prior, num_samples)
    keys = jax.random.split(key_node, len(non_prior_nodes))
    for key, node in zip(keys, non_prior_nodes):
        node_inputs = get_node_inputs(node, samples)
        # Nodes with parents need vmapping when batched to avoid shape issues.
        node_sample = sample_forward_node(node, key, num_samples, node_inputs)
        samples[node.name] = node_sample
    return samples


def sum_log_probs(log_prob_dict):
    return sum(log_prob_dict.values())


def get_namedtuple_value(field, named_tuple, default=None):
    # TODO change name to get_namedtuple_value_or_distribution
    if isinstance(named_tuple, Distribution):
        return named_tuple
    else:
        return getattr(named_tuple, field, default)


def _apply_priors(priors, inverse_samples):
    samples, log_prob = {}, {}
    for prior in priors:
        state = prior.apply(get_namedtuple_value(prior.name, inverse_samples))
        samples[prior.name] = get_namedtuple_value(prior.name, state.sample)
        log_prob[prior.name] = state.log_prob.sum()
    return samples, log_prob


def _apply(priors, non_priors, inverse_samples):
    samples, log_prob = _apply_priors(priors, inverse_samples)
    for node in non_priors:
        node_inputs = [samples[edge.name] for edge in node.edges]
        node_sample = get_namedtuple_value(node.name, inverse_samples)
        state = node.apply(node_sample, *node_inputs)
        log_prob[node.name] = state.log_prob.sum()
        samples[node.name] = get_namedtuple_value(node.name, state.sample)
    return samples, log_prob


def get_latent_nodes(nodes):
    return [node for node in nodes if node.sample_inverse is not None]


def PGM(inputs, outputs, name):
    nodes = search_nodes(outputs)
    _validate_unique_names(nodes)
    dag = DAG([node.name for node in nodes], get_edges(nodes), name)
    sorted_names = dag.sort_topologically()
    Sample = SampleType(sorted_names)
    non_priors = get_non_root_nodes(nodes, sorted_names, dag.root_nodes())
    latent_nodes = get_latent_nodes(nodes)
    latent_names = [node.name for node in latent_nodes]
    Latent = SampleType(latent_names)

    def sample_inverse(key, num_samples=1):
        return Latent(**_sample_inverse(inputs, latent_nodes, key, num_samples))

    def sample_forward(key, num_samples=1):
        return Sample(**_sample_forward(inputs, non_priors, key, num_samples))

    def apply(inverse_samples):
        sample, log_prob = _apply(inputs, non_priors, inverse_samples)
        return NodeState(Sample(**sample), log_prob, sum_log_probs(log_prob))

    metadata = PGMMetadata(nodes, inputs, non_priors, latent_nodes)
    return Variable(
        apply, sample_forward, sample_inverse, name, [], None, metadata
    )
