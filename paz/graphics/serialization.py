import json
import jax.numpy as jp
from paz.graphics.types import PointLight, Material, Pattern, Shape, Group


def serialize(serializable):
    """Recursively converts namedtuples and arrays into a JSON-serializable"""
    if isinstance(serializable, jp.ndarray):
        return serializable.tolist()
    if isinstance(serializable, tuple) and hasattr(serializable, "_asdict"):
        dictionary = serializable._asdict()
        return {k: serialize(v) for k, v in dictionary.items()}
    if isinstance(serializable, list):
        return [serialize(item) for item in serializable]
    return serializable


def _reconstruct_light(data):
    """Reconstructs a PointLight object from a dictionary."""
    if data is not None:
        lights = PointLight(
            intensity=jp.array(data["intensity"]),
            position=jp.array(data["position"]),
        )
    else:
        lights = None
    return lights


def _reconstruct_material(data):
    """Reconstructs a Material object from a dictionary."""
    if data is None:
        return None
    return Material(
        color=jp.array(data["color"]),
        ambient=data["ambient"],
        diffuse=data["diffuse"],
        specular=data["specular"],
        shininess=data["shininess"],
    )


def _reconstruct_pattern(data):
    """Reconstructs a Pattern object from a dictionary."""
    if data is None:
        return None
    return Pattern(
        transform=jp.array(data["transform"]),
        type=data["type"],
        image=jp.array(data["image"]),
    )


def _reconstruct_shape(data):
    """Reconstructs a Shape object from a dictionary."""
    if data is not None:
        shape = Shape(
            transform=jp.array(data["transform"]),
            type=data["type"],
            material=_reconstruct_material(data["material"]),
            pattern=_reconstruct_pattern(data["pattern"]),
        )
    else:
        shape = None
    return shape


def _reconstruct_group(data):
    """Reconstructs a Group object from a dictionary."""
    if data is not None:
        shapes = [_reconstruct_shape(shape) for shape in data["shapes"]]
        parent_array = jp.array(data["parent_array"])
        group = Group(shapes=shapes, parent_array=parent_array)
    else:
        group = None
    return group


def save(filepath, **serializables):
    """Serializes a complete scene setup to a single JSON file."""
    json_compatible_data = {k: serialize(v) for k, v in serializables.items()}
    with open(filepath, "w") as f:
        json.dump(json_compatible_data, f, indent=4)


def load(filepath):
    """Deserializes a complete scene setup from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    components = {}
    for key, value in data.items():
        if key == "shapes":
            if isinstance(value, dict) and "parent_array" in value:
                components[key] = _reconstruct_group(value)
            elif isinstance(value, list):
                components[key] = [_reconstruct_shape(s) for s in value]
            elif isinstance(value, dict):
                components[key] = _reconstruct_shape(value)
            else:
                raise TypeError(f"Unknown structure for '{key}': {value}")
        elif key == "lights":
            components[key] = [_reconstruct_light(light) for light in value]
        elif key == "camera_pose":
            components[key] = jp.array(value)
        else:
            components[key] = value

    return components
