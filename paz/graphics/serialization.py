import json
import pathlib
import jax
import jax.numpy as jp
from paz.graphics.types import Material, Pattern, Shape, Group, Scene


def to_json(scene_node_field):
    if isinstance(scene_node_field, jp.ndarray):
        return scene_node_field.tolist()
    if isinstance(scene_node_field, tuple) and hasattr(
        scene_node_field, "_asdict"
    ):
        return {k: to_json(v) for k, v in scene_node_field._asdict().items()}
    if isinstance(scene_node_field, list):
        return [to_json(item) for item in scene_node_field]
    return scene_node_field


def save(filepath, scene):
    path = pathlib.Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Filepath must have a .json extension.")
    with open(filepath, "w") as f:
        json.dump(to_json(scene), f, indent=4)


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


# def build_material(data):
#     return Material(**to_array_if_vector(data))


# def build_pattern(data):
#     return Pattern(**to_array_if_vector(data))


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
    # if node_type == "Group":
    if "shapes" in node_data:
        return build_group(node_data)
    elif "material" in node_data:
        return build_shape(node_data)
    else:
        raise TypeError(f"Node is not a valid Shape or Group: {node_data}")


def build_scene(data):
    nodes = [build_node(node) for node in data.get("nodes", [])]
    return Scene(nodes, jp.array(data.get("parent_array", [])))


def load(filepath):
    """Deserializes a scene from a JSON file."""
    with open(filepath, "r") as filedata:
        data = json.load(filedata)
    return build_scene(data)
