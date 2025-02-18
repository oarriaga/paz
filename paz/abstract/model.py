from collections import namedtuple
from typing import Callable

from paz.abstract.tree import Tree


Processor = namedtuple("Pocessor", ["apply", "name", "edges"])


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
    non_root_names = list(set(node_names) - set(prior_names))
    non_root_nodes = []
    for node in nodes:
        if node.name in non_root_names:
            non_root_nodes.append(node)
    return non_root_nodes


def build_type(node_names):
    if isinstance(node_names, list):
        Type = namedtuple("Type", [name for name in node_names])
    else:
        raise ValueError("'node_names' must be a list of strings")
    return Type


def get_namedtuple_value(field, named_tuple, default=None):
    return getattr(named_tuple, field, default)


def get_values_by_key(keys, dictionary):
    return [dictionary[key] for key in keys]


def get_node_inputs(node, dictionary):
    return get_values_by_key(get_edge_names(node), dictionary)


def get_edge_names(node):
    return [edge.name for edge in node.edges]


def apply_root_nodes(root_nodes, inverse_samples):
    samples = {}
    for node in root_nodes:
        state = node.apply(get_namedtuple_value(node.name, inverse_samples))
        samples[node.name] = get_namedtuple_value(node.name, state.sample)
    return samples


def apply_model(root_nodes, non_root_nodes, inverse_samples):
    samples = apply_root_nodes(root_nodes, inverse_samples)
    for node in non_root_nodes:
        node_inputs = get_node_inputs(node, inverse_samples)
        node_sample = get_namedtuple_value(node.name, inverse_samples)
        state = node.apply(node_sample, *node_inputs)
        samples[node.name] = get_namedtuple_value(node.name, state.sample)
    return samples


def Node(name, apply_fn):
    if not isinstance(apply_fn, Callable):
        raise ValueError(f"Input {apply_fn} must be a callable")

    Sample = SampleType([name])
    edges = []

    def apply(inverse_sample, *args):
        forward_sample = bijector(inverse_sample)
        distribution = apply_fn(*args)
        log_prob = distribution.log_prob(forward_sample)
        return NodeState(Sample(forward_sample), log_prob.sum())

    def call(*args):
        for arg in args:
            edges.append(arg)
        return Processor(apply, name, edges)

    return call


def Model(inputs, outputs, name):
    nodes = search_nodes(outputs)
    tree = Tree([node.name for node in nodes], get_edges(nodes), name)
    sorted_names = tree.sort_topologically()
    OutputType = build_type(sorted_names)
    non_root_nodes = get_non_root_nodes(nodes, sorted_names, tree.root_nodes())

    def apply(inverse_samples):
        sample = apply_model(inputs, non_root_nodes, inverse_samples)
        return OutputType(**sample)

    return Processor(apply, name, [])


import jax

image = paz.Input("image")
x = Node(paz.image.normalize)(image)
x = Node(paz.image.rgb_to_grayscale)(x)
preprocess = jax.vmap(Model(image, x))

probs = paz.Input("probs")
names = Node(paz.lock(paz.classes.to_name, class_names))(classes)
postprocess = jax.vmap(Model(probs, [probs, names]))

batch_images = Input("batch_images")
x = Node(preprocess)(batch_images)
x = model(x)
x = Node(postprocess)(x)
x = Node(paz.draw.boxes)(x)
model = Model(batch_images, x)
