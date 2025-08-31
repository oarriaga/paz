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


def compute_num_shapes(scene):
    total_shapes = 0
    for shape_or_group in scene.nodes:
        if isinstance(shape_or_group, paz.graphics.Shape):
            total_shapes = total_shapes + 1
        elif isinstance(shape_or_group, paz.graphics.Group):
            total_shapes = total_shapes + len(shape_or_group.shapes)
        else:
            raise TypeError("Scene elements must be 'Shape' or 'Group' types.")
    return total_shapes


def prepare_mask(mask, scene):
    """Prepares user mask to match flat scene."""

    if mask is None:
        flat_mask = jp.ones(compute_num_shapes(scene), dtype=bool)
    else:
        if len(mask) != len(scene.nodes):
            raise ValueError("Mask length must match top-level scene elements.")
        flat_mask = expand_mask(mask, scene)

    return flat_mask


def validate_scene(scene):
    if not isinstance(scene, paz.graphics.Scene):
        raise TypeError("'scene' must be a paz.graphics.Scene type.")
    for node in scene.nodes:
        if not isinstance(node, (paz.graphics.Shape, paz.graphics.Group)):
            raise TypeError("'scene' elements must be 'Shape' or 'Group' type.")


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


def get_groups(scene):
    groups = []
    for shape_or_group in scene.nodes:
        if isinstance(shape_or_group, paz.graphics.Group):
            groups.append(shape_or_group)
    return groups


def get_shapes(scene):
    shapes = []
    for shape_or_group in scene.nodes:
        if isinstance(shape_or_group, paz.graphics.Shape):
            shapes.append(shape_or_group)
    return shapes


def prepare_group(group):
    if not isinstance(group.shapes, list):
        raise TypeError("'group.shapes' must be a list of Shapes.")

    transformed_shapes = []
    for shape in group.shapes:
        transform = group.transform @ shape.transform
        transformed_shapes.append(shape._replace(transform=transform))

    if not transformed_shapes:
        return None

    return prepare_shapes(transformed_shapes)


def prepare_shapes(shapes):
    if isinstance(shapes, list):
        if not all(isinstance(shape, paz.graphics.Shape) for shape in shapes):
            raise TypeError("All elements must be Shape objects.")
    else:
        raise TypeError("'shapes' must be a Shape or a list of Shapes.")

    return shapes


def prepare_groups(groups):
    return [prepare_group(group) for group in groups]


def prepare_scene(scene):
    validate_scene(scene)
    flat_scene = []
    for group in get_groups(scene):
        shapes = prepare_group(group)
        flat_scene.extend(shapes)

    for shape in get_shapes(scene):
        flat_scene.append(shape)

    for flat_shape in flat_scene:
        if not isinstance(flat_shape, paz.graphics.Shape):
            raise TypeError("All elements in flat_scene must be Shape objects.")

    return flat_scene


def compile(scene, lights, mask):
    return (
        prepare_scene(scene),
        prepare_lights(lights),
        prepare_mask(mask, scene),
    )
