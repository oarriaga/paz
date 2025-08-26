# paz/graphics/serialization.py

import json
import pathlib
import jax.numpy as jp
from paz.graphics.types import PointLight, Material, Pattern, Shape, Group


def serialize(obj):
    if isinstance(obj, jp.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return {k: serialize(v) for k, v in obj._asdict().items()}
    if isinstance(obj, list):
        return [serialize(item) for item in obj]
    return obj


def _reconstruct_light(data):
    if data is None:
        return None
    return PointLight(
        intensity=jp.array(data["intensity"]),
        position=jp.array(data["position"]),
    )


def _reconstruct_material(data):
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
    if data is None:
        return None
    return Pattern(
        transform=jp.array(data["transform"]),
        type=data["type"],
        image=jp.array(data["image"]),
    )


def _reconstruct_shape(data):
    if data is None:
        return None
    return Shape(
        transform=jp.array(data["transform"]),
        type=data["type"],
        material=_reconstruct_material(data["material"]),
        pattern=_reconstruct_pattern(data["pattern"]),
    )


def _reconstruct_group(data):
    if data is None:
        return None
    return Group(
        shapes=[_reconstruct_shape(s) for s in data["shapes"]],
        parent_array=jp.array(data["parent_array"]),
    )


def _reconstruct_component(key, value):
    """Helper to reconstruct a single component based on its key."""
    if key in ["shapes", "group"]:
        if isinstance(value, dict) and "parent_array" in value:
            return _reconstruct_group(value)
        elif isinstance(value, list):
            return [_reconstruct_shape(s) for s in value]
        else:
            raise TypeError(f"Unknown structure for '{key}': {value}")
    elif key == "lights":
        return [_reconstruct_light(light) for light in value]
    elif key == "camera_pose":
        return jp.array(value)
    else:
        return value


def save(filepath, **serializables):
    """Serializes a complete scene setup to a single JSON file."""
    path = pathlib.Path(filepath)
    if path.suffix.lower() != ".json":
        raise ValueError("Filepath must have a .json extension.")

    json_compatible_data = {
        key: serialize(value) for key, value in serializables.items()
    }
    with open(filepath, "w") as f:
        json.dump(json_compatible_data, f, indent=4)


def load(filepath):
    """
    Deserializes a scene from a JSON file.
    If the file contains a single object, it is returned directly.
    If it contains multiple objects, a dictionary is returned.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    if len(data) == 1:
        key, value = list(data.items())[0]
        return _reconstruct_component(key, value)
    else:
        return {
            key: _reconstruct_component(key, value)
            for key, value in data.items()
        }
