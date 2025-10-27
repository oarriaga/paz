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


def prepare_mask(mask, num_shapes):
    """Prepares user mask to match flat scene."""

    if mask is None:
        flat_mask = jp.ones(num_shapes, dtype=bool)
    else:
        if len(mask) != num_shapes:
            raise ValueError("Mask length must match top-level scene elements.")
        flat_mask = mask  # TODO expand_mask using recursion

    return flat_mask


def expand_mask(mask, scene):
    """Expands user mask to match flat scene."""
    expanded_mask = []
    for mask_value, shape_or_group in zip(mask, scene.nodes):
        if isinstance(shape_or_group, paz.graphics.Shape):
            expanded_mask.append(mask_value)
        elif isinstance(shape_or_group, paz.graphics.Group):
            num_shapes_in_group = len(shape_or_group.shapes)
            expanded_mask.extend([mask_value] * num_shapes_in_group)
        else:
            raise TypeError(f"Node {shape_or_group} is not 'Shape' or 'Group'.")
    return jp.array(expanded_mask, dtype=bool)


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
        # prepare_scene(scene),
        flat_scene,
        prepare_lights(lights),
        prepare_mask(mask, len(flat_scene)),
    )
