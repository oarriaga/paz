import jax.numpy as jp
import jax
import paz
from paz.graphics import PointLight, Shape


def prepare_lights(lights):
    if isinstance(lights, PointLight):
        processed_lights = [lights]
    elif isinstance(lights, list):
        if not all(isinstance(light, PointLight) for light in lights):
            raise TypeError("All elements must be PointLight objects.")
        processed_lights = lights
    else:
        raise TypeError("'lights' must be a PointLight or list of PointLights.")
    return processed_lights


def relative_to_world(relative_transforms, parent_array):
    """Converts a scene graph with relative transforms to world transforms."""

    def compute_world_transform(world_transforms, node_arg):
        parent_arg = parent_array[node_arg]
        relative_transform = relative_transforms[node_arg]
        parent_transform = jax.lax.cond(
            parent_arg == -1,
            lambda: jp.eye(4),
            lambda: world_transforms[parent_arg],
        )
        world_transform = parent_transform @ relative_transform
        updated_transforms = world_transforms.at[node_arg].set(world_transform)
        return updated_transforms, None

    sorted_indices = paz.abstract.tree.sort_topologically(parent_array)
    initial_world_transforms = jp.zeros_like(relative_transforms)
    world_transforms, _ = jax.lax.scan(
        compute_world_transform,
        initial_world_transforms,
        jp.array(sorted_indices),
    )
    return world_transforms


def compute_num_shapes(scene):
    # TODO expand to count for multiple levels of groups
    total_shapes = 0
    for shape_or_group in scene.nodes:
        if isinstance(shape_or_group, Shape):
            total_shapes = total_shapes + 1
        elif isinstance(shape_or_group, paz.graphics.Group):
            total_shapes = total_shapes + len(shape_or_group.shapes)
        else:
            raise TypeError("Scene elements must be 'Shape' or 'Group' types.")
    return total_shapes


def expand_mask(mask, scene):
    """Expands user mask to match flat scene."""
    expanded_mask = []
    for mask_value, shape_or_group in zip(mask, scene):
        if isinstance(shape_or_group, Shape):
            expanded_mask.append(mask_value)
        elif isinstance(shape_or_group, paz.graphics.Group):
            num_shapes_in_group = len(shape_or_group.shapes)
            expanded_mask.extend([mask_value] * num_shapes_in_group)
    return jp.array(expanded_mask, dtype=bool)


def prepare_mask(mask, scene):
    """Prepares user mask to match flat scene."""

    if mask is None:
        flat_mask = jp.ones(compute_num_shapes(scene), dtype=bool)
    else:
        if len(mask) != len(scene.nodes):
            raise ValueError("Mask length must match top-level scene elements.")
        flat_mask = expand_mask(mask, scene)

    return flat_mask


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


def validate_scene(scene):
    if not isinstance(scene, paz.graphics.Scene):
        raise TypeError("'scene' must be a paz.graphics.Scene type.")
    for node in scene.nodes:
        if not isinstance(node, (paz.graphics.Shape, paz.graphics.Group)):
            raise TypeError("'scene' elements must be 'Shape' or 'Group' type.")
    return True


def prepare_group(group):
    if isinstance(group.shapes, list):
        if not all(isinstance(x, paz.graphics.Shape) for x in group.shapes):
            raise TypeError("All group.shapes elements must be Shape objects.")
        batched_shapes = prepare_shapes(group.shapes)
    else:
        raise TypeError("'group.shapes' must be a list of Shapes.")
    return batched_shapes

    # poses = relative_to_world(batched_shapes.transform, group.parent_array)
    # return batched_shapes._replace(transform=poses)


def prepare_groups(groups):
    return [prepare_group(group) for group in groups]


def prepare_shapes(shapes):
    if isinstance(shapes, list):
        if not all(isinstance(shape, paz.graphics.Shape) for shape in shapes):
            raise TypeError("All elements must be Shape objects.")
        if len(shapes) == 1:
            batched_shapes = paz.graphics.shapes.expand(shapes[0])
        else:
            batched_shapes = paz.graphics.shapes.merge(*shapes)
    else:
        raise TypeError("'shapes' must be a Shape or a list of Shapes.")

    return batched_shapes


def flatten_scene(scene):
    batched_shapes = []
    batched_shapes.extend(prepare_groups(get_groups(scene)))
    shapes = get_shapes(scene)
    if shapes:
        merged_standalone_shapes = prepare_shapes(shapes)
        batched_shapes.append(merged_standalone_shapes)
    return paz.graphics.shapes.concatenate(*batched_shapes)


def compile(scene, lights, mask):
    """Prepares the entire scene, lights, and mask for the core renderer."""
    validate_scene(scene)
    flat_scene = flatten_scene(scene)
    parent_array = jp.array(scene.parent_array)
    transforms = relative_to_world(flat_scene.transform, parent_array)
    flat_scene = flat_scene._replace(transform=transforms)
    return flat_scene, prepare_lights(lights), prepare_mask(mask, scene)
