import jax.numpy as jp
import paz
from paz.graphics import PointLight


def prepare_lights(lights):
    if isinstance(lights, paz.graphics.PointLight):
        processed_lights = [lights]
    elif isinstance(lights, list):
        if not all(isinstance(light, PointLight) for light in lights):
            raise TypeError("All elements must be PointLight objects.")
        processed_lights = lights
    else:
        raise TypeError("'lights' must be a PointLight or list of PointLights.")
    return processed_lights


def prepare_mask(mask, num_shapes, scene):
    """Prepares user mask to match flat scene."""

    if mask is None:
        flat_mask = jp.ones(num_shapes, dtype=bool)
    else:
        if len(mask) != len(scene.nodes):
            raise ValueError("Mask length must match top-level scene elements.")
        flat_mask = expand_mask(mask, scene)
    return flat_mask


def expand_mask_node(mask_value, node):
    if isinstance(node, paz.graphics.Shape):
        return [mask_value]
    elif isinstance(node, paz.graphics.Group):
        group_mask = []
        for shape_or_group in node.shapes:
            group_mask.extend(expand_mask_node(mask_value, shape_or_group))
        return group_mask
    else:
        raise ValueError(f"Invalid node type: {type(node)}")


def expand_mask(mask, scene):
    expanded_mask = []
    for mask_value, node in zip(mask, scene.nodes):
        expanded_mask.extend(expand_mask_node(mask_value, node))
    return jp.array(expanded_mask)


def flatten(node, accumulated_transform):
    if isinstance(node, paz.graphics.Shape):
        return [node._replace(transform=accumulated_transform @ node.transform)]
    elif isinstance(node, paz.graphics.Group):
        children, child_transform = [], accumulated_transform @ node.transform
        for shape_or_group in node.shapes:
            children.extend(flatten(shape_or_group, child_transform))
        return children
    else:
        raise ValueError(f"Invalid node type{type(node)}.")


def flatten_scene(scene):
    flat_scene = []
    for node in scene.nodes:
        flat_scene.extend(flatten(node, jp.eye(4)))
    return flat_scene


def compile(scene, lights, mask):
    flat_scene = flatten_scene(scene)
    return (
        flat_scene,
        prepare_lights(lights),
        prepare_mask(mask, len(flat_scene), scene),
    )
