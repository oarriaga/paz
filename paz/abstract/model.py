from collections import namedtuple
from typing import Callable

from paz.abstract.tree import Tree
from paz.abstract.node import build_type, Node
import paz


def search_nodes(output_nodes):
    tree_nodes, queue = [], output_nodes
    while len(queue) != 0:
        node = queue.pop(0)
        queue.extend(node.edges)
        if node not in [tree_nodes]:
            tree_nodes.append(node)
    return tree_nodes


def build_edge(edge_name, node_name):
    source_name = edge_name
    target_name = node_name
    edge = [source_name, target_name]
    return edge


def get_edges(nodes):
    edges = []
    for node in nodes:
        for edge in node.edges:
            edges.append(build_edge(edge.name, node.name))
    return edges


def get_values_by_key(keys, dictionary):
    return [dictionary[key] for key in keys]


def get_edge_names(node):
    return [edge.name for edge in node.edges]


def get_node_inputs(node, dictionary):
    return get_values_by_key(get_edge_names(node), dictionary)


def get_namedtuple_value(field, named_tuple, default=None):
    return getattr(named_tuple, field, default)


def apply_non_root_node(node, all_outputs):
    node_inputs = get_node_inputs(node, all_outputs)
    node_output = node.call(*node_inputs)
    return get_namedtuple_value(node.name, node_output)


def apply_root_nodes(root_nodes, *model_inputs):
    return {node.name: arg for node, arg in zip(root_nodes, model_inputs)}


def filter_dictionary_by_keys(input_dictionary, keys):
    output_dictionary = {}
    for key in keys:
        output_dictionary[key] = input_dictionary[key]
    return output_dictionary


def get_non_root_nodes(nodes, sorted_node_names, input_node_names):
    non_root_nodes = []
    for node_name_to_search in sorted_node_names:
        if node_name_to_search in input_node_names:
            continue
        else:
            for node in nodes:
                if node.name == node_name_to_search:
                    non_root_nodes.append(node)
    return non_root_nodes


def Model(inputs, outputs, name):
    output_names = [output_node.name for output_node in outputs]
    nodes = search_nodes(outputs)
    tree = Tree([node.name for node in nodes], get_edges(nodes), name)
    sorted_names = tree.sort_topologically()
    root_names = [node.name for node in inputs]
    if set(tree.root_nodes()) != set(root_names):
        raise ValueError(f"Not all inputs {inputs} are root {root_names}")
    Type = build_type(output_names)
    non_root_nodes = [node for node in nodes if node.name not in root_names]
    sorted_root_nodes = get_non_root_nodes(nodes, sorted_names, root_names)

    def call(*args):
        all_outputs = apply_root_nodes(inputs, *args)
        for node in sorted_root_nodes:
            all_outputs[node.name] = apply_non_root_node(node, all_outputs)
            # TODO remove unsued outputs for memory efficiency
        model_outputs = filter_dictionary_by_keys(all_outputs, output_names)
        return Type(**model_outputs)

    return call
