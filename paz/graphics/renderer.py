from collections import namedtuple

import jax
import jax.numpy as jp

import paz

from paz.graphics.composite import (
    compute_scene_hit_mask,
    find_closest_intersection_args,
    postprocess,
    take_closest,
)

# TODO retune: measured best under 6k faces, but 2048 is 1.75x faster
# at 82k, so this is wrong for the room-scale meshes ahead.
FACE_CHUNK_SIZE = 128
SHADOW_ORIGIN_EPSILON = 1e-5
SHADOW_SELF_HIT_EPSILON = 1e-5
BOUNCE_ORIGIN_EPSILON = 1e-2

TRIANGLE_HIT_NAMES = "hit_mask depth points normals eyes albedo primitive"
TriangleHit = namedtuple("TriangleHit", TRIANGLE_HIT_NAMES.split())

STATE_NAMES = "color depth hit_mask throughput active_mask "
STATE_NAMES += "refractive_index rays"
RenderState = namedtuple("RenderState", STATE_NAMES.split())

MATERIAL_NAMES = "reflectivities transparencies refractive_indices"
HitMaterial = namedtuple("HitMaterial", MATERIAL_NAMES.split())

Surfaces = namedtuple("Surfaces", ["points", "normals", "eyes"])


def render(
    shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size,
    shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    compiled = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)
    trace_args = compiled, shadows, num_bounces, face_chunk_size, chunk_size
    trace = paz.lock(trace_chunks, *trace_args)
    tile_step = paz.lock(render_tile_step, shape, y_FOV, pose, tiles, trace)
    images, depths = scan_tiles(shape, tiles, tile_step)
    image = assemble_tiles(shape, tiles, images)
    depth = assemble_tiles(shape, tiles, depths)[..., 0]
    return image, depth


def render_masks(
    shape, y_FOV, pose, scene, lights, depth, tiles, chunk_size,
    num_objects=None, shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    if num_objects is None:
        num_objects = len(scene.nodes)
    min_depth, max_depth = depth
    masks = []
    for object_arg in range(num_objects):
        mask = build_object_mask(len(scene.nodes), object_arg)
        args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
        args += shadows, shadow_mask, num_bounces, face_chunk_size
        _, depth_image = render(*args)
        soft = paz.depth.to_soft_mask(depth_image, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def build_object_mask(num_nodes, object_arg):
    return jp.zeros((num_nodes,), dtype=bool).at[object_arg].set(True)


def scan_tiles(shape, tiles, tile_step):
    H, W = shape
    H_tiles, W_tiles = tiles
    paz.graphics.mesh.assert_exact_tile_side(H, H_tiles)
    paz.graphics.mesh.assert_exact_tile_side(W, W_tiles)
    coordinates = paz.graphics.mesh.make_tile_coordinates(H_tiles, W_tiles)
    return jax.lax.scan(tile_step, None, coordinates)[1]


def render_tile_step(carry, tile_arg, shape, y_FOV, pose, tiles, trace):
    H, W = shape
    H_tiles, W_tiles = tiles
    camera_to_world = jp.linalg.inv(pose)
    tile_args = H, W, H_tiles, W_tiles, y_FOV, camera_to_world
    rays = paz.graphics.mesh.build_tile_rays(*tile_args, tile_arg)
    hit_mask, depth, color = trace(rays)
    tile_H, tile_W = H // H_tiles, W // W_tiles
    post_args = hit_mask, depth, color, pose, rays, tile_H, tile_W
    image, depth = postprocess(*post_args)
    return carry, (image, jp.expand_dims(depth, -1))


def assemble_tiles(shape, tiles, images):
    H, W = shape
    H_tiles, W_tiles = tiles
    return paz.graphics.mesh.assemble(H, W, H_tiles, W_tiles, images)


def trace_chunks(rays, compiled, shadows, num_bounces, face_chunk, chunk_size):
    ray_chunks = split_ray_chunks(rays, chunk_size)
    step_args = compiled, shadows, num_bounces, face_chunk
    trace_step = paz.lock(trace_chunk_step, *step_args)
    hit_mask, depth, color = jax.lax.scan(trace_step, None, ray_chunks)[1]
    return flatten_chunk_results(hit_mask, depth, color, len(rays[0]))


def trace_chunk_step(carry, rays, compiled, shadows, num_bounces, face_chunk):
    args = rays, compiled, shadows, num_bounces, face_chunk
    return carry, trace_bounces(*args)


def split_ray_chunks(rays, chunk_size):
    origins, directions = rays
    origins = pad_to_chunks(origins, chunk_size)
    directions = pad_to_chunks(directions, chunk_size)
    num_chunks = origins.shape[0] // chunk_size
    shape = num_chunks, chunk_size, 3
    return origins.reshape(shape), directions.reshape(shape)


def pad_to_chunks(array, chunk_size):
    remainder = array.shape[0] % chunk_size
    if remainder == 0:
        padded = array
    else:
        padding = jp.repeat(array[-1:], chunk_size - remainder, axis=0)
        padded = jp.concatenate([array, padding], axis=0)
    return padded


def flatten_chunk_results(hit_mask, depth, color, num_rays):
    hit_mask = flatten_chunk_array(hit_mask, num_rays)
    depth = flatten_chunk_array(depth, num_rays)
    color = flatten_chunk_array(color, num_rays)
    return hit_mask, depth, color


def flatten_chunk_array(array, num_rays):
    shape = (-1,) + array.shape[2:]
    return array.reshape(shape)[:num_rays]


def trace_bounces(rays, compiled, shadows, bounces, face_chunk):
    state = initialize_state(rays)
    bounce = paz.lock(bounce_step, compiled, shadows, face_chunk)
    for step_arg in range(bounces):
        state = bounce(state, step_arg)
    return state.hit_mask, state.depth, state.color


def initialize_state(rays):
    num_rays = rays[0].shape[0]
    color = jp.zeros((num_rays, 3))
    depth = jp.full((num_rays,), paz.graphics.FARAWAY)
    hit_mask = jp.zeros((num_rays,), dtype=bool)
    throughput = jp.ones((num_rays, 3))
    active_mask = jp.ones((num_rays,), dtype=bool)
    refractive_index = jp.ones((num_rays,))
    fields = color, depth, hit_mask, throughput, active_mask
    return RenderState(*fields, refractive_index, rays)


def bounce_step(state, bounce, compiled, shadows, face_chunk):
    triangle_hit = compute_triangle_hit(compiled, state.rays, face_chunk)
    hit_masks, depths, surfaces = intersect(compiled, state.rays, triangle_hit)
    closest = gather_closest(hit_masks, depths, surfaces)
    state = update_first_hit(state, closest, bounce)
    state = update_active_mask(state, closest)
    color_args = compiled, closest, surfaces, shadows, triangle_hit
    colors = compute_hit_colors(*color_args)
    return update_state(state, compiled, closest, colors)


def update_first_hit(state, closest, bounce):
    if bounce == 0:
        depth, hit_mask = closest.depth, closest.hit_mask
    else:
        depth, hit_mask = state.depth, state.hit_mask
    return state._replace(depth=depth, hit_mask=hit_mask)


def update_active_mask(state, closest):
    active_mask = state.active_mask & closest.hit_mask
    return state._replace(active_mask=active_mask)


def compute_triangle_hit(compiled, rays, face_chunk):
    if compiled.triangles is None:
        triangle_hit = None
    else:
        triangle_hit = build_triangle_hit(compiled, rays, face_chunk)
    return triangle_hit


def build_triangle_hit(compiled, rays, face_chunk):
    triangles = compiled.triangles
    args = triangles, rays, face_chunk
    result = paz.graphics.mesh.intersect_triangles(*args)
    hit_mask, depth, points, normals, eyes, face_index, u, v = result
    primitive = triangles.primitive_index[face_index]
    hit_mask = jp.logical_and(hit_mask, compiled.triangle_mask[primitive])
    depth = jp.where(hit_mask, depth, paz.graphics.FARAWAY)
    albedo_args = triangles, face_index, u, v
    albedo = paz.graphics.albedo.compute_triangle_albedo(*albedo_args)
    args = hit_mask, depth, points, normals, eyes, albedo, primitive
    return TriangleHit(*args)


def intersect(compiled, rays, triangle_hit):
    rows = []
    if len(compiled.shapes) > 0:
        rows.append(intersect_shapes(compiled.shapes, rays, compiled.mask))
    if triangle_hit is not None:
        rows.append(build_triangle_row(triangle_hit))
    joined = tuple(jp.concatenate(fields, axis=0) for fields in zip(*rows))
    hit_masks, depths, points, normals, eyes = joined
    return hit_masks, depths, Surfaces(points, normals, eyes)


def intersect_shapes(shapes, rays, mask):
    merged = paz.graphics.shapes.field_merge(shapes, ["transform", "type"])
    intersect_fun = paz.lock(paz.graphics.shapes.intersect, *rays)
    hit_masks, depths, points, normals, eyes = jax.vmap(intersect_fun)(merged)
    hit_masks = jp.where(jp.expand_dims(mask, 1), hit_masks, False)
    return hit_masks, depths, points, normals, eyes


def build_triangle_row(triangle_hit):
    depth = jp.expand_dims(triangle_hit.depth, -1)
    fields = triangle_hit.hit_mask, depth, triangle_hit.points
    fields += triangle_hit.normals, triangle_hit.eyes
    return tuple(jp.expand_dims(field, 0) for field in fields)


def gather_closest(hit_masks, depths, surfaces):
    indices = find_closest_intersection_args(hit_masks, depths)
    fields = hit_masks, depths, *surfaces
    closest = tuple(take_closest(field, indices) for field in fields)
    return paz.graphics.Hit(*closest, indices)


def compute_hit_colors(compiled, closest, surfaces, shadows, triangle_hit):
    if len(compiled.shapes) == 0:
        colors = compute_triangle_colors(compiled, triangle_hit)
    elif triangle_hit is None:
        colors = compute_shape_colors(compiled, closest, surfaces, shadows)
    else:
        args = compiled, closest, surfaces, shadows, triangle_hit
        colors = blend_hit_colors(*args)
    return colors


def blend_hit_colors(compiled, closest, surfaces, shadows, triangle_hit):
    # TODO a mixed scene shades both paths for every ray and throws one
    # away. Joining them before selection needs color_with_shadows to
    # return rows instead of selecting inside its per-light scan.
    shape_colors = compute_shape_colors(compiled, closest, surfaces, shadows)
    triangle_colors = compute_triangle_colors(compiled, triangle_hit)
    is_triangle = closest.primitive_index == len(compiled.shapes)
    is_triangle = jp.expand_dims(is_triangle, -1)
    return jp.where(is_triangle, triangle_colors, shape_colors)


def compute_shape_colors(compiled, closest, surfaces, shadows):
    num_shapes = len(compiled.shapes)
    indices = jp.minimum(closest.primitive_index, num_shapes - 1)
    surfaces = slice_surfaces(surfaces, 0, num_shapes)
    if shadows:
        colors = color_with_shadows(compiled, closest, surfaces, indices)
    else:
        colors = color_without_shadow(compiled, surfaces, indices)
    return colors


def compute_triangle_colors(compiled, triangle_hit):
    materials = compiled.triangles.materials
    material = gather_triangle_material(materials, triangle_hit.primitive)
    shader = select_shader(materials)
    colors = jp.zeros_like(triangle_hit.albedo)
    for light in compiled.lights:
        args = triangle_hit.albedo, material, triangle_hit.points
        args += triangle_hit.normals, triangle_hit.eyes, light
        colors = colors + shader.compute_colors(*args)
    return colors


def gather_triangle_material(materials, primitive):
    material = jax.tree.map(lambda field: field[primitive], materials)
    return jax.tree.map(expand_scalar_field, material)


def expand_scalar_field(field):
    if field.ndim == 1:
        field = jp.expand_dims(field, -1)
    return field


def select_shader(material):
    if isinstance(material, paz.graphics.CookTorranceMaterial):
        shader = paz.graphics.cook_torrance
    else:
        shader = paz.graphics.phong
    return shader


def color_without_shadow(compiled, surfaces, indices):
    colors = []
    lights = paz.graphics.shapes.merge(*compiled.lights)
    for group, start_arg, final_arg in iterate_shape_groups(compiled.shapes):
        group_surfaces = slice_surfaces(surfaces, start_arg, final_arg)
        albedo = compute_group_albedo(group, group_surfaces.points)
        shader = select_shader(group.material)
        axes = 0, 0, 0, 0, 0, None
        color_per_light = jax.vmap(shader.compute_colors, axes)
        color = jax.vmap(color_per_light, (None, None, None, None, None, 0))
        args = albedo, group.material, *group_surfaces, lights
        colors.append(jp.sum(color(*args), axis=0))
    return take_closest(jp.concatenate(colors, axis=0), indices)


def iterate_shape_groups(shapes):
    start_arg = 0
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        final_arg = start_arg + len(group)
        yield paz.graphics.shapes.merge(*group), start_arg, final_arg
        start_arg = final_arg


def slice_surfaces(surfaces, start_arg, final_arg):
    points = surfaces.points[start_arg:final_arg]
    normals = surfaces.normals[start_arg:final_arg]
    eyes = surfaces.eyes[start_arg:final_arg]
    return Surfaces(points, normals, eyes)


def compute_group_albedo(group, points):
    compute_albedo = paz.graphics.albedo.compute_shape_albedo
    return jax.vmap(compute_albedo)(group, group.material, points)


def color_with_shadows(compiled, closest, surfaces, indices):
    colors = jp.zeros((len(surfaces.points[0]), 3))
    lights = paz.graphics.shapes.merge(*compiled.lights)
    body = paz.lock(scan_light_step, compiled, closest, surfaces, indices)
    return jax.lax.scan(body, colors, lights)[0]


def scan_light_step(colors, light, compiled, closest, surfaces, indices):
    args = compiled, closest, surfaces, indices, light
    return colors + compute_light_colors(*args), None


def compute_light_colors(compiled, closest, surfaces, indices, light):
    directions, distance = compute_light_directions(light, closest.point)
    occlusion_args = compiled, closest, indices, directions, distance
    is_shadow = compute_light_occlusion(*occlusion_args)
    color_args = compiled.shapes, light, surfaces, is_shadow
    return take_closest(compute_shadowed_colors(*color_args), indices)


def compute_light_directions(light, points):
    vector = light.position - points
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def compute_light_occlusion(compiled, closest, indices, directions, distance):
    origins = compute_shadow_ray_origins(closest.point, closest.normal)
    shadow_args = compiled.shapes, origins, directions
    hit_masks, depths, _, _, _, casters = intersect_shadow_groups(*shadow_args)
    masks = resolve_shadow_masks(compiled, hit_masks)
    depth_args = masks, depths, casters, indices, closest.normal, directions
    masks, depths = select_shadow_depths(*depth_args)
    return compute_soft_occlusion(masks, depths, distance)


def compute_shadow_ray_origins(points, normals):
    over_point, _ = compute_surface_points(points, normals)
    return over_point


def compute_surface_points(point, normal):
    over_point = point + normal * SHADOW_ORIGIN_EPSILON
    under_point = point - normal * SHADOW_ORIGIN_EPSILON
    return over_point, under_point


def intersect_shadow_groups(shapes, origins, directions):
    intersect_all = paz.graphics.shapes.intersect_all
    intersect_group = jax.vmap(paz.lock(intersect_all, origins, directions))
    rows = []
    for group, start_arg, final_arg in iterate_shape_groups(shapes):
        indices = jp.arange(start_arg, final_arg)
        rows.append((*intersect_group(group), indices))
    return tuple(jp.concatenate(fields, axis=0) for fields in zip(*rows))


def resolve_shadow_masks(compiled, hit_masks):
    transparencies = compute_transparencies(compiled.shapes)
    masks = jp.where(jp.expand_dims(compiled.mask, 1), hit_masks, False)
    masks = hide_transparent_shapes(masks, transparencies > 0.0)
    if compiled.shadow_mask is None:
        resolved = masks
    else:
        resolved = hide_non_casting_shapes(masks, compiled.shadow_mask)
    return resolved


def compute_transparencies(shapes):
    return jp.array([shape.material.transparency for shape in shapes])


def hide_transparent_shapes(shadow_masks, is_transparent):
    return jp.where(jp.expand_dims(is_transparent, 1), False, shadow_masks)


def hide_non_casting_shapes(shadow_masks, shadow_mask):
    return jp.where(jp.expand_dims(shadow_mask, 1), shadow_masks, False)


def select_shadow_depths(
    hit_masks, depths, casters, receivers, normals, directions
):
    same_shape = casters[:, None] == receivers[None, :]
    front_args = same_shape, normals, directions
    front_side_hits = compute_front_side_shadow_mask(*front_args)
    root_args = hit_masks, depths, same_shape, front_side_hits
    valid_roots = compute_valid_roots(*root_args)
    depths = jp.where(valid_roots, depths, paz.graphics.FARAWAY)
    return jp.any(valid_roots, axis=1), jp.min(depths, axis=1)


def compute_front_side_shadow_mask(same_shape, normals, directions):
    front_side = paz.algebra.dot(normals, directions) >= 0.0
    return jp.logical_and(same_shape, front_side[None, :])


def compute_valid_roots(hit_masks, depths, same_shape, front_side_hits):
    thresholds = compute_shadow_depth_thresholds(same_shape)
    valid_roots = depths > thresholds[:, None, :]
    valid_roots = jp.logical_and(valid_roots, depths < paz.graphics.FARAWAY)
    valid_roots = jp.logical_and(jp.expand_dims(hit_masks, 1), valid_roots)
    return jp.logical_and(valid_roots, ~front_side_hits[:, None, :])


def compute_shadow_depth_thresholds(same_shape):
    return jp.where(same_shape, SHADOW_SELF_HIT_EPSILON, paz.graphics.EPSILON)


def compute_soft_occlusion(hit_masks, depths, light_lengths, slope=0.01):
    closest_depths = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_depths = jp.min(closest_depths, axis=0)
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    blockers = closest_depths <= light_lengths
    blocker_mask = jp.logical_and(scene_hit_mask, blockers)
    difference = light_lengths - closest_depths
    occlusion = jax.nn.sigmoid(slope * difference)
    return jp.where(blocker_mask, occlusion, 0.0)


def compute_shadowed_colors(shapes, light, surfaces, is_shadow):
    colors = []
    for group, start_arg, final_arg in iterate_shape_groups(shapes):
        group_surfaces = slice_surfaces(surfaces, start_arg, final_arg)
        albedo = compute_group_albedo(group, group_surfaces.points)
        shader = select_shader(group.material)
        axes = 0, 0, 0, 0, 0, None, None
        color = jax.vmap(shader.compute_colors_with_shadow, axes)
        args = albedo, group.material, *group_surfaces, light, is_shadow
        colors.append(color(*args))
    return jp.concatenate(colors, axis=0)


def update_state(state, compiled, closest, hit_colors):
    material = compute_material_properties(compiled, closest.primitive_index)
    color_args = state.color, state.throughput, state.active_mask, hit_colors
    color_args += material.reflectivities, material.transparencies
    color = accumulate_color(*color_args)
    new_rays, n_2, reflectance = compute_bounce(state, closest, material)
    args = new_rays, n_2, material, reflectance
    return apply_bounce_update(state._replace(color=color), *args)


def compute_material_properties(compiled, hit_shape_args):
    values = collect_material_values(compiled)
    gathered = tuple(jp.array(row)[hit_shape_args] for row in values)
    return HitMaterial(*gathered)


def collect_material_values(compiled):
    reflectivities, transparencies, refractive_indices = [], [], []
    for shape in compiled.shapes:
        reflectivities.append(shape.material.reflective)
        transparencies.append(shape.material.transparency)
        refractive_indices.append(shape.material.refractive_index)
    if compiled.triangles is not None:
        reflectivities.append(0.0)
        transparencies.append(0.0)
        refractive_indices.append(1.0)
    return reflectivities, transparencies, refractive_indices


def accumulate_color(
    colors, throughput, active_mask, hit_colors, reflectivities, transparencies
):
    weights = jp.maximum(1.0 - reflectivities - transparencies, 0.0)
    weights = jp.expand_dims(weights, -1)
    active_mask = jp.expand_dims(active_mask, -1)
    return colors + (throughput * active_mask * weights * hit_colors)


def compute_bounce(state, closest, material):
    args = state.rays[1], closest.normal, state.refractive_index
    terms = compute_refraction_terms(*args, material.refractive_indices)
    normal, eye, n_1, n_2, n_ratio = terms
    reflectance = compute_reflectance(normal, eye, n_1, n_2)
    ray_args = normal, eye, n_ratio, closest.point, material.transparencies
    return compute_new_rays(*ray_args), n_2, reflectance


def compute_refraction_terms(directions, normal, n_1, refractive_indices):
    eye = -directions
    normal, is_inside = flip_normal_if_inside(eye, normal)
    # TODO why 1.0 hardcoded
    n_2 = jp.where(is_inside, 1.0, refractive_indices)
    return normal, eye, n_1, n_2, n_1 / n_2


def flip_normal_if_inside(eye, normal):
    is_inside = jp.sum(normal * eye, axis=-1) < 0.0
    return jp.where(jp.expand_dims(is_inside, -1), -normal, normal), is_inside


def compute_reflectance(normal, eye, n_1, n_2):
    cosines = compute_transmission_cosines(eye, normal, n_1 / n_2)
    cos_incident, sin_transmit_squared, cos_transmit = cosines
    cos = jp.where(n_1 > n_2, cos_transmit, cos_incident)
    base_reflectance = ((n_1 - n_2) / (n_1 + n_2)) ** 2
    grazing = (1.0 - cos) ** 5
    reflectance = base_reflectance + (1.0 - base_reflectance) * grazing
    return jp.where(sin_transmit_squared > 1.0, 1.0, reflectance)


def compute_transmission_cosines(eye, normal, n_ratio):
    cos_incident = jp.sum(eye * normal, axis=-1)
    sin_transmit_squared = (n_ratio**2) * (1.0 - (cos_incident**2))
    cos_transmit = jp.sqrt(jp.maximum(0.0, 1.0 - sin_transmit_squared))
    return cos_incident, sin_transmit_squared, cos_transmit


def compute_new_rays(normal, eye, n_ratio, point, transparencies):
    is_transparent = transparencies > 0.0
    do_reflect = jp.expand_dims(~is_transparent, -1)
    reflection = paz.graphics.geometry.reflect(-eye, normal)
    refraction = compute_refractive_direction(eye, normal, n_ratio)
    direction = jp.where(do_reflect, reflection, refraction)
    lower_point, upper_point = displace_by_normal(point, normal)
    origin = jp.where(do_reflect, upper_point, lower_point)
    return origin, paz.algebra.normalize(direction)


def compute_refractive_direction(eye, normal, n_ratio):
    args = eye, normal, n_ratio
    cos_incident, _, cos_transmit = compute_transmission_cosines(*args)
    inside_vector = -eye * jp.expand_dims(n_ratio, -1)
    up_weight = n_ratio * cos_incident - cos_transmit
    return jp.expand_dims(up_weight, -1) * normal + inside_vector


def displace_by_normal(point, normal):
    upper_point = point + normal * BOUNCE_ORIGIN_EPSILON
    lower_point = point - normal * BOUNCE_ORIGIN_EPSILON
    return lower_point, upper_point


def apply_bounce_update(state, new_rays, n_2, material, reflectance):
    is_transparent = material.transparencies > 0.0
    is_reflective = material.reflectivities > 0.0
    factor = compute_bounce_factor(material, reflectance)
    factor = jp.where(is_transparent & (reflectance >= 1.0), 1.0, factor)
    throughput = state.throughput * jp.expand_dims(factor, -1)
    active_mask = state.active_mask & (is_transparent | is_reflective)
    index_args = state, n_2, is_transparent, reflectance
    refractive_index = update_refractive_index(*index_args)
    args = throughput, active_mask, refractive_index, new_rays
    return replace_bounce_state(state, *args)


def compute_bounce_factor(material, reflectance):
    is_transparent = material.transparencies > 0.0
    is_reflective = material.reflectivities > 0.0
    transparent_factor = material.transparencies * (1.0 - reflectance)
    reflective_factor = jp.where(is_reflective, material.reflectivities, 0.0)
    return jp.where(is_transparent, transparent_factor, reflective_factor)


def update_refractive_index(state, n_2, is_transparent, reflectance):
    update_mask = is_transparent & (reflectance < 1.0)
    return jp.where(update_mask, n_2, state.refractive_index)


def replace_bounce_state(state, throughput, active_mask, index, rays):
    state = state._replace(throughput=throughput, active_mask=active_mask)
    return state._replace(refractive_index=index, rays=rays)
