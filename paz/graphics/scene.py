import jax
import jax.numpy as jp
import paz
from paz.graphics import PointLight, AreaLight, Shape
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
    if isinstance(lights, (PointLight, AreaLight)):
        lights = [lights]
    elif not isinstance(lights, list):
        raise TypeError("'lights' must be a Light or list of Lights.")
    return expand_lights(lights)


def expand_lights(lights):
    expanded = []
    for light in lights:
        if isinstance(light, AreaLight):
            expanded.extend(expand_area_light(light))
        else:
            expanded.append(light)
    return expanded


def expand_area_light(light):
    cells = light.usteps * light.vsteps
    intensity = light.intensity / cells
    positions = build_area_light_positions(light)
    return [PointLight(intensity, position) for position in positions]


def build_area_light_positions(light):
    cells = light.usteps * light.vsteps
    u = jp.tile(jp.arange(light.usteps), light.vsteps)[:, None]
    v = jp.repeat(jp.arange(light.vsteps), light.usteps)[:, None]
    offsets = build_jitter_offsets(light, cells)
    u_step = light.edge1 / light.usteps
    v_step = light.edge2 / light.vsteps
    positions = light.corner + u_step * (u + offsets[:, :1])
    positions = positions + v_step * (v + offsets[:, 1:])
    return list(positions)


def build_jitter_offsets(light, cells):
    if light.key is None:
        return jp.full((cells, 2), 0.5)
    return jax.random.uniform(light.key, (cells, 2))


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
    if isinstance(node, (paz.graphics.Shape, paz.graphics.Mesh)):
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
    if isinstance(node, (paz.graphics.Shape, paz.graphics.Mesh)):
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


def sort_by_group(shapes, mask, shadow_mask):
    """Sorts shapes, mask and shadow_mask to make groups contiguous."""
    groups = paz.graphics.shapes.group_by_pattern_size(shapes)
    sorted_shapes = []
    for group in groups.values():
        sorted_shapes.extend(group)
    ID_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    args = [ID_to_arg[id(shape)] for shape in sorted_shapes]
    order = jp.array(args, dtype=jp.int32)

    mask = mask[order]
    if shadow_mask is not None:
        shadow_mask = shadow_mask[order]
    return sorted_shapes, mask, shadow_mask


def compile(scene, lights, mask, shadow_mask=None):
    flat_scene = flatten_scene(scene)
    lights = prepare_lights(lights)
    mask = prepare_mask(mask, len(flat_scene), scene)

    if shadow_mask is not None:
        shadow_mask = prepare_mask(shadow_mask, len(flat_scene), scene)

    shape_args = select_shapes(flat_scene, mask, shadow_mask)
    shapes, shape_mask, shadow_mask = sort_by_group(*shape_args)
    meshes, triangle_mask = select_meshes(flat_scene, mask)
    args = shapes, build_triangles(meshes), lights, shape_mask
    return paz.graphics.CompiledScene(*args, shadow_mask, triangle_mask)


def select_shapes(flat_scene, mask, shadow_mask):
    args = [arg for arg, node in enumerate(flat_scene) if is_shape(node)]
    shapes = [flat_scene[arg] for arg in args]
    args = jp.array(args, dtype=jp.int32)
    if shadow_mask is not None:
        shadow_mask = shadow_mask[args]
    return shapes, mask[args], shadow_mask


def select_meshes(flat_scene, mask):
    args = [arg for arg, node in enumerate(flat_scene) if not is_shape(node)]
    meshes = [flat_scene[arg] for arg in args]
    return meshes, mask[jp.array(args, dtype=jp.int32)]


def is_shape(node):
    return isinstance(node, paz.graphics.Shape)


def build_triangles(meshes):
    if len(meshes) == 0:
        triangles = None
    else:
        triangles = build_mesh_triangles(meshes)
    return triangles


def build_mesh_triangles(meshes):
    vertices = [bake_vertices(mesh) for mesh in meshes]
    vertex_uvs = [build_vertex_uvs(mesh) for mesh in meshes]
    vertex_colors = [mesh.vertex_colors for mesh in meshes]
    args = jp.concatenate(vertices), build_offset_faces(meshes)
    args += jp.concatenate(vertex_uvs), jp.concatenate(vertex_colors)
    args += build_primitive_index(meshes), stack_materials(meshes)
    args += (stack_patterns(meshes),)
    return paz.graphics.Triangles(*args)


def bake_vertices(mesh):
    return paz.algebra.transform_points(mesh.transform, mesh.vertices)


def build_offset_faces(meshes):
    faces, offset = [], 0
    for mesh in meshes:
        faces.append(mesh.faces + offset)
        offset = offset + len(mesh.vertices)
    return jp.concatenate(faces)


def build_vertex_uvs(mesh):
    if mesh.vertex_uvs is None:
        vertex_uvs = jp.zeros((len(mesh.vertices), 2))
    else:
        vertex_uvs = mesh.vertex_uvs
    return vertex_uvs


def build_primitive_index(meshes):
    indices = []
    for index, mesh in enumerate(meshes):
        indices.append(jp.full(len(mesh.faces), index, dtype=jp.int32))
    return jp.concatenate(indices)


def stack_materials(meshes):
    materials = [mesh.material for mesh in meshes]
    return jax.tree.map(lambda *args: jp.stack(args), *materials)


def stack_patterns(meshes):
    patterns = [build_mesh_pattern(mesh) for mesh in meshes]
    shapes = {pattern.image.shape for pattern in patterns}
    if len(shapes) > 1:
        raise ValueError("Mesh pattern images must all have equal shape.")
    return jax.tree.map(lambda *args: jp.stack(args), *patterns)


def build_mesh_pattern(mesh):
    if mesh.pattern is None:
        pattern = paz.graphics.Pattern()
    else:
        pattern = mesh.pattern
    return pattern
