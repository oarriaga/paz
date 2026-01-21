from paz.abstract.dag import DAG
from paz.inference.latent import Latent
from paz.inference.observable import Observable
from paz.inference.pgm.builder import search_nodes
from paz.inference.prior import Prior
from paz.inference.serialization.serializable import build_distribution_fn

from .spec import (
    _decode_kwargs,
    _deserialize_bijector,
    _deserialize_distribution_obj,
    _encode_kwargs,
    _serialize_bijector,
    _serialize_distribution_obj,
)


def _serialize_graph(output_nodes, arrays):
    nodes = search_nodes(output_nodes)
    edges = [[edge.name, node.name] for node in nodes for edge in node.edges]
    dag = DAG([node.name for node in nodes], edges, "pgm")
    order = dag.sort_topologically()
    name_to_node = {node.name: node for node in nodes}
    node_specs = [_serialize_node(name_to_node[name], arrays) for name in order]
    return {"order": order, "nodes": node_specs}


def _serialize_node(node, arrays):
    edges = [edge.name for edge in node.edges]
    if node.distribution is not None:
        spec = _serialize_distribution_obj(node.distribution, arrays, node.name)
        bijector = _serialize_bijector(
            node.bijector, arrays, f"{node.name}_bijector"
        )
        return {
            "type": "prior",
            "name": node.name,
            "edges": edges,
            "distribution": spec,
            "bijector": bijector,
        }
    distribution_fn = node.distribution_fn
    fn_spec = _serialize_distribution_fn(distribution_fn, arrays, node.name)
    bijector = _serialize_bijector(
        node.bijector, arrays, f"{node.name}_bijector"
    )
    node_type = "latent" if node.sample_inverse is not None else "observable"
    return {
        "type": node_type,
        "name": node.name,
        "edges": edges,
        "distribution_fn": fn_spec,
        "bijector": bijector,
    }


def _build_graph(spec, arrays):
    order = spec["order"]
    node_specs = {node["name"]: node for node in spec["nodes"]}
    nodes = {}
    for name in order:
        nodes[name] = _build_node(node_specs[name], nodes, arrays)
    return nodes


def _build_node(spec, nodes, arrays):
    node_type = spec["type"]
    name = spec["name"]
    edges = [nodes[edge] for edge in spec.get("edges", [])]
    if node_type == "prior":
        distribution = _deserialize_distribution_obj(spec["distribution"], arrays)
        bijector = _deserialize_bijector(spec["bijector"], arrays)
        return Prior(distribution, bijector=bijector, name=name)
    distribution_fn = _deserialize_distribution_fn(spec["distribution_fn"], arrays)
    bijector = _deserialize_bijector(spec["bijector"], arrays)
    if node_type == "latent":
        return Latent(distribution_fn, bijector=bijector, name=name)(*edges)
    return Observable(distribution_fn, name=name)(*edges)


def _serialize_distribution_fn(distribution_fn, arrays, prefix):
    spec = getattr(distribution_fn, "_paz_spec", None)
    if spec is None:
        raise ValueError(
            "This object contains unserializable callables. Use "
            "a serializable spec."
        )
    kwargs = _encode_kwargs(spec["kwargs"], arrays, prefix)
    return {"fn_id": spec["fn_id"], "kwargs": kwargs}


def _deserialize_distribution_fn(spec, arrays):
    kwargs = _decode_kwargs(spec["kwargs"], arrays)
    return build_distribution_fn(spec["fn_id"], **kwargs)
