from collections import Counter, namedtuple

from paz.abstract.dag import DAG
from paz.inference.latent_space import build_latent_space
from paz.inference.types import PGMMetadata, SampleType


PGMContext = namedtuple(
    "PGMContext",
    [
        "inputs",
        "non_priors",
        "latent_nodes",
        "latent_nodes_sorted",
        "output_nodes",
        "observable_names",
        "latent_space",
        "Sample",
        "Latent",
        "metadata",
    ],
)


def build_pgm_context(inputs, outputs, name):
    nodes = search_nodes(outputs)
    _validate_unique_names(nodes)
    edges = [
        [edge.name, node.name] for node in nodes for edge in node.edges
    ]
    dag = DAG([node.name for node in nodes], edges, name)
    sorted_names = dag.sort_topologically()
    Sample = SampleType(sorted_names)
    non_priors = get_non_root_nodes(nodes, sorted_names, dag.root_nodes())
    latent_nodes = get_latent_nodes(nodes)
    name_to_node = {node.name: node for node in nodes}
    latent_names = [node.name for node in latent_nodes]
    latent_name_set = set(latent_names)
    latent_nodes_sorted = [
        name_to_node[name] for name in sorted_names if name in latent_name_set
    ]
    latent_space = build_latent_space(latent_nodes_sorted)
    output_nodes = list(outputs)
    observable_nodes = get_observable_nodes(nodes)
    observable_names = [node.name for node in observable_nodes]
    Latent = SampleType(latent_names)
    metadata = PGMMetadata(
        nodes, inputs, non_priors, latent_nodes, output_nodes, observable_nodes
    )
    return PGMContext(
        inputs,
        non_priors,
        latent_nodes,
        latent_nodes_sorted,
        output_nodes,
        observable_names,
        latent_space,
        Sample,
        Latent,
        metadata,
    )


def search_nodes(output_nodes):
    tree_nodes, queue = [], list(output_nodes)
    while len(queue) != 0:
        node = queue.pop(0)
        queue.extend(node.edges)
        if node not in tree_nodes:
            tree_nodes.append(node)
    return tree_nodes


def get_non_root_nodes(nodes, sorted_names, prior_names):
    non_prior_names = set(sorted_names) - set(prior_names)
    name_to_node = {node.name: node for node in nodes}
    return [
        name_to_node[name] for name in sorted_names if name in non_prior_names
    ]


def get_latent_nodes(nodes):
    return [node for node in nodes if node.sample_inverse is not None]


def get_observable_nodes(nodes):
    return [node for node in nodes if node.sample_inverse is None]


def _validate_unique_names(nodes):
    name_counts = Counter(node.name for node in nodes)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        duplicates.sort()
        raise ValueError(
            "PGM node names must be unique. "
            f"Duplicates: {', '.join(duplicates)}"
        )
