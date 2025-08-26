import jax.numpy as jp
import jax
import paz
from paz.graphics import PointLight, Shape


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


def prepare_group(group):
    if isinstance(group.shapes, list):
        if not all(isinstance(x, paz.graphics.Shape) for x in group.shapes):
            raise TypeError("All group.shapes elements must be Shape objects.")
        batched_shapes = prepare_shapes(group.shapes)
    else:
        raise TypeError("'group.shapes' must be a list of Shapes.")

    poses = relative_to_world(batched_shapes.transform, group.parent_array)
    return batched_shapes._replace(transform=poses)


def prepare_groups(groups):
    return [prepare_group(group) for group in groups]


def prepare_lights(lights):
    is_single_light = isinstance(lights, PointLight)
    is_light_list = isinstance(lights, list)

    if is_single_light:
        processed_lights = [lights]
    elif is_light_list:
        if not all(isinstance(light, PointLight) for light in lights):
            raise TypeError("All elements must be PointLight objects.")
        processed_lights = lights
    else:
        raise TypeError("'lights' must be a PointLight or list of PointLights.")
    return processed_lights


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


def prepare_mask(mask, scene_elements):
    """
    Prepares a user-provided mask to match the flattened scene structure.
    If the mask is None, it creates an all-True mask.
    """
    if not isinstance(scene_elements, list):
        scene_elements = [scene_elements]

    if mask is None:
        total_shapes = 0
        for element in scene_elements:
            if isinstance(element, Shape):
                total_shapes = total_shapes + 1
            elif isinstance(element, paz.graphics.Group):
                total_shapes = total_shapes + len(element.parent_array)
        return jp.ones(total_shapes, dtype=bool)

    if len(mask) != len(scene_elements):
        raise ValueError("Mask length must match top-level scene elements.")

    expanded_mask = []
    for mask_value, element in zip(mask, scene_elements):
        if isinstance(element, Shape):
            expanded_mask.append(mask_value)
        elif isinstance(element, paz.graphics.Group):
            num_shapes_in_group = len(element.parent_array)
            expanded_mask.extend([mask_value] * num_shapes_in_group)
    return jp.array(expanded_mask, dtype=bool)


def split_shapes_and_groups(scene_elements):
    if not isinstance(scene_elements, list):
        scene_elements = [scene_elements]
    shapes, groups = [], []
    for scene_element in scene_elements:
        if isinstance(scene_element, paz.graphics.Shape):
            shapes.append(scene_element)
        elif isinstance(scene_element, paz.graphics.Group):
            groups.append(scene_element)
        else:
            raise TypeError("All elements must be of type Shape or Group.")
    return shapes, groups


def prepare_scene(scene_elements):
    shapes, groups = split_shapes_and_groups(scene_elements)
    processed_groups = prepare_groups(groups)
    all_batched_components = []
    all_batched_components.extend(processed_groups)
    if shapes:
        merged_standalone_shapes = prepare_shapes(shapes)
        all_batched_components.append(merged_standalone_shapes)
    scene = paz.graphics.shapes.concatenate(*all_batched_components)
    return scene


def compile(scene, lights, mask):
    """Prepares the entire scene, lights, and mask for the core renderer."""
    processed_lights = prepare_lights(lights)
    processed_mask = prepare_mask(mask, scene)
    processed_scene = prepare_scene(scene)
    num_shapes = paz.graphics.shapes.get_num_shapes(processed_scene)
    if num_shapes != len(processed_mask):
        raise ValueError(
            f"Mismatch after compilation: Scene has {num_shapes} shapes, "
            f"but mask has {len(processed_mask)} elements."
        )
    return processed_scene, processed_lights, processed_mask
