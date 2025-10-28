import jax
from paz.inference import NodeState, Variable, SampleType, Distribution
from paz.abstract.tree import Tree


def search_nodes(output_nodes):
    tree_nodes, queue = [], output_nodes
    while len(queue) != 0:
        node = queue.pop(0)
        queue.extend(node.edges)
        if node not in [tree_nodes]:
            tree_nodes.append(node)
    return tree_nodes


def get_edges(nodes):
    edges = []
    for node in nodes:
        for edge in node.edges:
            source = edge.name
            target = node.name
            edges.append([source, target])
    return edges


def get_non_root_nodes(nodes, node_names, prior_names):
    non_prior_names = list(set(node_names) - set(prior_names))
    non_prior_nodes = []
    for node in nodes:
        if node.name in non_prior_names:
            non_prior_nodes.append(node)
    return non_prior_nodes


def get_non_leaf_names(node_names, leaf_names):
    return list(set(node_names) - set(leaf_names))


def get_non_leaf_nodes(sorted_nodes, node_names, leaf_names):
    non_leaf_names = get_non_leaf_names(node_names, leaf_names)
    non_leaf_nodes = []
    for node in sorted_nodes:
        if node.name in non_leaf_names:
            non_leaf_nodes.append(node)
    return non_leaf_nodes


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


def _sample_inverse(prior_nodes, non_root_nodes, key, num_samples):
    key_prior, key_node = jax.random.split(key)
    samples = _sample_inverse_priors(prior_nodes, key_prior, num_samples)
    keys = jax.random.split(key_node, len(non_root_nodes))
    for key, node in zip(keys, non_root_nodes):
        node_inputs = get_node_inputs(node, samples)
        node_inverse_sample = node.sample_inverse(key, 1, *node_inputs)
        samples[node.name] = node_inverse_sample
    return samples


def _sample_forward_priors(prior_nodes, key, num_samples):
    samples, keys = {}, jax.random.split(key, len(prior_nodes))
    for key, prior in zip(keys, prior_nodes):
        samples[prior.name] = prior.sample(key, num_samples)
    return samples


def _sample_forward(prior_nodes, non_root_nodes, key, num_samples):
    key_prior, key_node = jax.random.split(key)
    samples = _sample_forward_priors(prior_nodes, key_prior, num_samples)
    keys = jax.random.split(key_node, len(non_root_nodes))
    for key, node in zip(keys, non_root_nodes):
        node_inputs = get_node_inputs(node, samples)
        node_sample = node.sample(key, num_samples, *node_inputs)
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


def _get_node_inputs(node, namedtuple, default=None):
    node_inputs = []
    for edge in node.edges:
        node_inputs.append(getattr(namedtuple, edge.name, default))
    return node_inputs


def _apply(priors, non_priors, inverse_samples):
    samples, log_prob = _apply_priors(priors, inverse_samples)
    for node in non_priors:
        node_inputs = _get_node_inputs(node, inverse_samples)
        node_sample = get_namedtuple_value(node.name, inverse_samples)
        state = node.apply(node_sample, *node_inputs)
        log_prob[node.name] = state.log_prob.sum()
        samples[node.name] = get_namedtuple_value(node.name, state.sample)
    return samples, log_prob


def Model(inputs, outputs, name):
    nodes = search_nodes(outputs)
    tree = Tree([node.name for node in nodes], get_edges(nodes), name)
    sorted_names = tree.sort_topologically()
    Sample = SampleType(sorted_names)
    non_priors = get_non_root_nodes(nodes, sorted_names, tree.root_nodes())
    latent_nodes = get_non_leaf_nodes(nodes, sorted_names, tree.leaves())
    latent_names = get_non_leaf_names(sorted_names, tree.leaves())
    Latent = SampleType(latent_names)

    def sample_inverse(key, num_samples=1):
        return Latent(**_sample_inverse(inputs, latent_nodes, key, num_samples))

    def sample_forward(key, num_samples=1):
        return Sample(**_sample_forward(inputs, non_priors, key, num_samples))

    def apply(inverse_samples):
        sample, log_prob = _apply(inputs, non_priors, inverse_samples)
        return NodeState(Sample(**sample), log_prob, sum_log_probs(log_prob))

    return Variable(apply, sample_forward, sample_inverse, name, [], None)
