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
    if isinstance(node, Shape):
        name = type_to_name[node.type]
    elif isinstance(node, paz.graphics.CSG):
        operations = {paz.graphics.UNION: "Union"}
        operations[paz.graphics.INTERSECTION] = "Intersection"
        operations[paz.graphics.DIFFERENCE] = "Difference"
        name = operations[int(node.operation)]
    else:
        name = "Group   "
    return name


def get_tree_data(node, prefix, is_last):
    BOLD, GREEN, BLUE, RESET = "\033[1m", "\033[92m", "\033[94m", "\033[0m"
    name = _get_node_name(node)
    if isinstance(node, (paz.graphics.Group, paz.graphics.CSG)):
        label = f"{BOLD}{BLUE}{name}{RESET}"
    else:
        label = f"{GREEN}{name}{RESET}"
    connector = "└── " if is_last else "├── "
    main_string = f"{prefix}{connector}{label}"
    if isinstance(node, paz.graphics.CSG):
        data_string = ""
    else:
        data_string = _build_node_data_string(node, prefix, is_last)
    now_total_lines = [main_string + data_string]
    new_prefix = prefix + ("    " if is_last else "│   ")
    now_lines, now_child_count = [], 0
    if isinstance(node, (paz.graphics.Group, paz.graphics.CSG)):
        children = list_children(node)
        count = len(children)
        for child_arg, child in enumerate(children):
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
        expanded = [mask_value]
    else:
        expanded = expand_child_masks(mask_value, list_children(node))
    return expanded


# A nested mask holds one entry per child. A single value masks them all.
def expand_child_masks(mask_value, children):
    is_nested = isinstance(mask_value, (list, tuple))
    if not (is_nested or jp.ndim(mask_value) > 0):
        mask_value = [mask_value] * len(children)
    if len(mask_value) != len(children):
        raise ValueError("Nested mask length must match the node children.")
    expanded = []
    for child_mask, child in zip(mask_value, children):
        expanded.extend(expand_mask_node(child_mask, child))
    return expanded


def list_children(node):
    if isinstance(node, (paz.graphics.Shape, paz.graphics.Mesh)):
        children = []
    elif isinstance(node, paz.graphics.Group):
        children = node.shapes
    elif isinstance(node, paz.graphics.CSG):
        children = [node.shape_A, node.shape_B]
    else:
        raise ValueError(f"Invalid node type: {type(node)}")
    return children


def expand_mask(mask, scene):
    expanded_mask = []
    for mask_value, node in zip(mask, scene.nodes):
        expanded_mask.extend(expand_mask_node(mask_value, node))
    return jp.array(expanded_mask)


def flatten(node, accumulated_transform):
    if isinstance(node, (paz.graphics.Shape, paz.graphics.Mesh)):
        transform = accumulated_transform @ node.transform
        flat_nodes = [node._replace(transform=transform)]
    else:
        flat_nodes = flatten_children(node, accumulated_transform)
    return flat_nodes


def flatten_children(node, accumulated_transform):
    if isinstance(node, paz.graphics.Group):
        accumulated_transform = accumulated_transform @ node.transform
    flat_nodes = []
    for child in list_children(node):
        flat_nodes.extend(flatten(child, accumulated_transform))
    return flat_nodes


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

    mesh_args = select_meshes(flat_scene, mask, shadow_mask)
    meshes, triangle_mask, triangle_shadow_mask = mesh_args
    shape_args = select_shapes(flat_scene, mask, shadow_mask)
    shapes, shape_mask, shadow_mask = sort_by_group(*shape_args)
    args = shapes, build_triangles(meshes), lights, shape_mask
    args += shadow_mask, triangle_mask, triangle_shadow_mask
    solids = build_solids(scene, flat_scene, shapes)
    return paz.graphics.CompiledScene(*args, solids)


# One CSG tree per solid, with indices into `shapes` as leaves.
def build_solids(scene, flat_scene, shapes):
    ID_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    shape_args = [ID_to_arg.get(id(node)) for node in flat_scene]
    return find_solids(scene.nodes, shape_args)


def find_solids(nodes, shape_args):
    solids = []
    for node, node_args in split_shape_args(nodes, shape_args):
        if isinstance(node, paz.graphics.CSG):
            solids.append(build_tree(node, node_args))
        elif isinstance(node, paz.graphics.Group):
            solids.extend(find_solids(node.shapes, node_args))
    return solids


# Pairs each node with the shape indices of its own leaves.
def split_shape_args(nodes, shape_args):
    pairs, offset = [], 0
    for node in nodes:
        final = offset + count_leaves(node)
        pairs.append((node, shape_args[offset:final]))
        offset = final
    return pairs


def count_leaves(node):
    if isinstance(node, (paz.graphics.Shape, paz.graphics.Mesh)):
        count = 1
    else:
        count = sum(count_leaves(child) for child in list_children(node))
    return count


def build_tree(node, shape_args):
    if isinstance(node, paz.graphics.Shape):
        tree = shape_args[0]
    elif isinstance(node, paz.graphics.Mesh):
        raise ValueError("A Mesh cannot be a CSG operand.")
    else:
        pairs = split_shape_args(list_children(node), shape_args)
        trees = [build_tree(*pair) for pair in pairs]
        tree = join_trees(node, trees)
    return tree


# A Group operand is the union of its children.
def join_trees(node, trees):
    if isinstance(node, paz.graphics.CSG):
        operation = node.operation
    else:
        operation = paz.graphics.UNION
    if len(trees) == 0:
        raise ValueError("An empty Group cannot be a CSG operand.")
    joined = trees[0]
    for tree in trees[1:]:
        joined = paz.graphics.CSG(joined, tree, operation)
    return joined


def select_shapes(flat_scene, mask, shadow_mask):
    args = [arg for arg, node in enumerate(flat_scene) if is_shape(node)]
    shapes = [flat_scene[arg] for arg in args]
    indices = jp.array(args, dtype=jp.int32)
    if shadow_mask is not None:
        shadow_mask = shadow_mask[indices]
    return shapes, mask[indices], shadow_mask


def select_meshes(flat_scene, mask, shadow_mask):
    args = [arg for arg, node in enumerate(flat_scene) if not is_shape(node)]
    indices = jp.array(args, dtype=jp.int32)
    meshes = [flat_scene[arg] for arg in args]
    if shadow_mask is None:
        mesh_shadow_mask = None
    else:
        mesh_shadow_mask = shadow_mask[indices]
    return meshes, mask[indices], mesh_shadow_mask


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
    args += stack_patterns(meshes), build_corner_normals(meshes, vertices)
    return paz.graphics.Triangles(*args)


def bake_vertices(mesh):
    return paz.algebra.transform_points(mesh.transform, mesh.vertices)


def build_corner_normals(meshes, vertices):
    corner_normals = []
    for mesh, baked_vertices in zip(meshes, vertices):
        corner_normals.append(build_mesh_corner_normals(mesh, baked_vertices))
    return jp.concatenate(corner_normals)


# Corner normals stay unnormalized so the hit normalizes exactly once.
# paz.algebra.normalize divides by norm + 1e-5, so it is not idempotent.
def build_mesh_corner_normals(mesh, vertices):
    if mesh.vertex_normals is None:
        corner_normals = repeat_face_normals(vertices, mesh.faces)
    else:
        corner_normals = transform_vertex_normals(mesh)[mesh.faces]
    return corner_normals


def repeat_face_normals(vertices, faces):
    extract_points = paz.graphics.mesh.extract_points
    points_A, points_B, points_C = extract_points(vertices, faces)
    normals = jp.cross(points_B - points_A, points_C - points_A)
    return jp.repeat(jp.expand_dims(normals, 1), 3, axis=1)


def transform_vertex_normals(mesh):
    inverse_transpose = jp.linalg.inv(mesh.transform).T
    args = inverse_transpose, mesh.vertex_normals
    return paz.algebra.transform_vectors(*args)


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
    types = {type(material) for material in materials}
    if len(types) > 1:
        raise ValueError("Mesh materials must all be of one type.")
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
