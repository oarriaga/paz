import jax.numpy as jp
import paz
from paz.graphics import PointLight, Shape
from paz.graphics import SPHERE, CYLINDER, CONE, CUBE, PLANE


def _build_node_data_string(node, prefix, is_last):
    t1, t2, t3 = node.transform[:3, 3]
    t1 = f" {t1:.1f}" if t1 >= 0.0 else f"{t1:.1f}"
    t2 = f" {t2:.1f}" if t2 >= 0.0 else f"{t2:.1f}"
    t3 = f" {t3:.1f}" if t3 >= 0.0 else f"{t3:.1f}"
    BOLD, RED, RESET = "\033[1m", "\033[91m", "\033[0m"
    return f" {BOLD}{RED}t:{RESET}[{t1}{RED},{RESET} {t2}{RED},{RESET} {t3}]"


def _get_node_name(node):
    types = [SPHERE, CYLINDER, CONE, CUBE, PLANE]
    names = ["Sphere  ", "Cylinder", "Cone    ", "Cube    ", "Plane   "]
    type_to_name = dict(zip(types, names))
    return type_to_name[node.type] if isinstance(node, Shape) else "Group   "


def get_tree_data(node, prefix, is_last):
    BOLD, GREEN, BLUE, RESET = "\033[1m", "\033[92m", "\033[94m", "\033[0m"
    name = _get_node_name(node)
    if isinstance(node, paz.graphics.Group):
        label = f"{BOLD}{BLUE}{name}{RESET}"
    else:
        label = f"{GREEN}{name}{RESET}"
    connector = "└── " if is_last else "├── "
    main_string = f"{prefix}{connector}{label}"
    data_string = _build_node_data_string(node, prefix, is_last)
    now_total_lines = [main_string + data_string]
    new_prefix = prefix + ("    " if is_last else "│   ")
    now_lines, now_child_count = [], 0
    if isinstance(node, paz.graphics.Group):
        count = len(node.shapes)
        for child_arg, child in enumerate(node.shapes):
            is_child_last = child_arg == count - 1
            child_data = get_tree_data(child, new_prefix, is_child_last)
            new_lines, new_child_count = child_data
            now_lines.extend(new_lines)
            now_child_count = now_child_count + new_child_count

    if isinstance(node, paz.graphics.Shape):
        now_child_count = 1 + now_child_count
    return now_total_lines + now_lines, now_child_count


def show(scene):
    all_lines, total_nodes, count = [], 0, len(scene.nodes)
    for i, node in enumerate(scene.nodes):
        is_last_node = i == count - 1
        lines, node_count = get_tree_data(node, "", is_last_node)
        all_lines.extend(lines)
        total_nodes = total_nodes + node_count

    print("\n" + "=" * 50)
    BOLD, YELLOW, RESET = "\033[1m", "\033[93m", "\033[0m"
    print(f"      SCENE HIERARCHY ({BOLD}{YELLOW}Total Shapes:{RESET} {total_nodes})")  # fmt: skip
    print("=" * 50)
    for line in all_lines:
        print(line)
    print("=" * 50 + "\n")


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


def compute_bounces(shapes):
    for shape in shapes:
        if shape.material.reflective > 0.0 or shape.material.transparency > 0.0:
            return 5
    return 1


def compile(scene, lights, mask):
    flat_scene = flatten_scene(scene)
    return (
        flat_scene,
        prepare_lights(lights),
        prepare_mask(mask, len(flat_scene), scene),
    )
