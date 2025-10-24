import json
import pathlib
import jax.numpy as jp
from paz.graphics.types import Material, Pattern, Shape, Group, Scene
import paz


def is_namedtuple(scene_node_field):
    return (scene_node_field, tuple) and hasattr(scene_node_field, "_asdict")


def is_group(node_data):
    return "shapes" in node_data


def is_shape(node_data):
    return "material" in node_data


def is_scene(node_data):
    return "nodes" in node_data


def pattern_contains_image(pattern_node):
    is_spherical = pattern_node.type == paz.graphics.SPHERICAL_PATTERN
    is_planar = pattern_node.type == paz.graphics.PLANAR_PATTERN
    is_cylindrical = pattern_node.type == paz.graphics.CYLINDRICAL_PATTERN
    return is_spherical or is_planar or is_cylindrical


def to_json(scene_node_field, counter=[0], filepath=None):
    if isinstance(scene_node_field, jp.ndarray):
        return scene_node_field.tolist()
    if isinstance(scene_node_field, Pattern):
        if pattern_contains_image(scene_node_field):
            image = paz.image.denormalize(scene_node_field.image)
            if filepath is not None:
                image_filename = f"{filepath.stem}_pattern_{counter[0]}.png"
                image_path = filepath.with_name(image_filename)
            else:
                image_filename = f"image_pattern_{counter[0]}.png"
                image_path = image_filename
            paz.image.write(str(image_path), image)
            scene_node_field = scene_node_field._replace(image=image_filename)
            counter[0] += 1
        return {
            k: to_json(v, counter, filepath)
            for k, v in scene_node_field._asdict().items()
        }
    if is_namedtuple(scene_node_field):
        return {
            k: to_json(v, counter, filepath)
            for k, v in scene_node_field._asdict().items()
        }
    if isinstance(scene_node_field, list):
        return [to_json(item, counter, filepath) for item in scene_node_field]
    return scene_node_field


def save(filepath, component):
    path = pathlib.Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Filepath must have a .json extension.")
    with open(filepath, "w") as f:
        json.dump(to_json(component, counter=[0], filepath=path), f, indent=4)


def build_material(data):
    return Material(
        color=jp.array(data["color"]),
        ambient=data["ambient"],
        diffuse=data["diffuse"],
        specular=data["specular"],
        shininess=data["shininess"],
    )


# def build_pattern(data):
#     return Pattern(
#         transform=jp.array(data["transform"]),
#         type=data["type"],
#         image=jp.array(data["image"]),
#     )


def build_pattern(data, base_path=None):
    """Builds a Pattern, loading the image from disk if it's a path."""
    image_data = data["image"]
    if isinstance(image_data, str):
        if base_path:
            image_path = base_path / image_data
        else:
            image_path = image_data
        if not pathlib.Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = paz.image.load(str(image_path))
        image = paz.image.normalize(image)
    else:
        image = jp.array(image_data)
    transform = jp.array(data["transform"])
    return Pattern(transform=transform, type=data["type"], image=image)


def build_shape(data, base_path=None):
    return Shape(
        transform=jp.array(data["transform"]),
        type=data["type"],
        material=build_material(data.get("material")),
        pattern=build_pattern(data.get("pattern"), base_path),
    )


def build_group(data, base_path=None):
    shapes = [build_shape(shape, base_path) for shape in data["shapes"]]
    return Group(shapes=shapes, transform=jp.array(data["transform"]))


def build_node(node_data, base_path=None):
    if not isinstance(node_data, dict):
        raise TypeError(f"Data must be a dict, but got {type(node_data)}.")
    if is_group(node_data):
        return build_group(node_data, base_path)
    elif is_shape(node_data):
        return build_shape(node_data, base_path)
    else:
        raise TypeError(f"Node is not a valid Shape or Group: {node_data}")


def build_scene(data, base_path=None):
    nodes = [build_node(node, base_path) for node in data.get("nodes", [])]
    return Scene(nodes, jp.array(data.get("parent_array", [])))


def load(filepath):
    """Deserializes a scene from a JSON file."""
    path = pathlib.Path(filepath)
    base_path = path.parent
    with open(filepath, "r") as filedata:
        json_data = json.load(filedata)
    if is_scene(json_data):
        data = build_scene(json_data, base_path)
    elif is_group(json_data):
        data = build_group(json_data, base_path)
    elif is_shape(json_data):
        data = build_shape(json_data, base_path)
    else:
        raise TypeError("Data is not a valid Scene, Group or Shape.")
    return data
