import json
import pathlib
import jax
import jax.numpy as jp
from paz.graphics.types import Material, Pattern, Shape, Group, Scene


def is_namedtuple(scene_node_field):
    return (scene_node_field, tuple) and hasattr(scene_node_field, "_asdict")


def is_group(node_data):
    return "shapes" in node_data


def is_shape(node_data):
    return "material" in node_data


def is_scene(node_data):
    return "nodes" in node_data


def to_json(scene_node_field):
    if isinstance(scene_node_field, jp.ndarray):
        return scene_node_field.tolist()
    if is_namedtuple(scene_node_field):
        return {k: to_json(v) for k, v in scene_node_field._asdict().items()}
    if isinstance(scene_node_field, list):
        return [to_json(item) for item in scene_node_field]
    return scene_node_field


def save(filepath, component):
    path = pathlib.Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Filepath must have a .json extension.")
    with open(filepath, "w") as f:
        json.dump(to_json(component), f, indent=4)


def to_array_if_vector(data):
    def _to_array_if_vector(leaf):
        if isinstance(leaf, (list, tuple)):
            return jp.array(leaf)
        return leaf

    return jax.tree.map(_to_array_if_vector, data)


def build_material(data):
    return Material(
        color=jp.array(data["color"]),
        ambient=data["ambient"],
        diffuse=data["diffuse"],
        specular=data["specular"],
        shininess=data["shininess"],
    )


def build_pattern(data):
    return Pattern(
        transform=jp.array(data["transform"]),
        type=data["type"],
        image=jp.array(data["image"]),
    )


def build_shape(data):
    return Shape(
        transform=jp.array(data["transform"]),
        type=data["type"],
        material=build_material(data.get("material")),
        pattern=build_pattern(data.get("pattern")),
    )


def build_group(data):
    shapes = [build_shape(shape) for shape in data["shapes"]]
    return Group(shapes=shapes, transform=jp.array(data["transform"]))


def build_node(node_data):
    if not isinstance(node_data, dict):
        raise TypeError(f"Data must be a dict, but got {type(node_data)}.")
    if is_group(node_data):
        return build_group(node_data)
    elif is_shape(node_data):
        return build_shape(node_data)
    else:
        raise TypeError(f"Node is not a valid Shape or Group: {node_data}")


def build_scene(data):
    nodes = [build_node(node) for node in data.get("nodes", [])]
    return Scene(nodes, jp.array(data.get("parent_array", [])))


def load(filepath):
    """Deserializes a scene from a JSON file."""
    with open(filepath, "r") as filedata:
        json_data = json.load(filedata)
    if is_scene(json_data):
        data = build_scene(json_data)
    elif is_group(json_data):
        data = build_group(json_data)
    elif is_shape(json_data):
        data = build_shape(json_data)
    else:
        raise TypeError("Data is not a valid Scene, Group or Shape.")
    return data
