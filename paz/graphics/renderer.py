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

FACE_CHUNK_SIZE = 128
SHADOW_ORIGIN_EPSILON = 1e-5
SHADOW_SELF_HIT_EPSILON = 1e-5
# BOUNCE_ORIGIN_EPSILON = 3e-3
BOUNCE_ORIGIN_EPSILON = 1e-2
RENDER_NAMES = "shape y_FOV pose scene tiles chunk_size shadows "
RENDER_NAMES += "num_bounces face_chunk_size"
TRIANGLE_HIT_NAMES = "hit_mask depth points normals eyes albedo primitive"
STATE_NAMES = "color depth hit_mask throughput active_mask "
STATE_NAMES += "refractive_index rays"
SHADOW_COLOR_NAMES = "rays shapes lights indices mask shadow_mask "
SHADOW_COLOR_NAMES += "point normal points normals eyes"

RenderArgs = namedtuple("RenderArgs", RENDER_NAMES.split())
TriangleHit = namedtuple("TriangleHit", TRIANGLE_HIT_NAMES.split())
RenderState = namedtuple("RenderState", STATE_NAMES.split())
ShadowColorArgs = namedtuple("ShadowColorArgs", SHADOW_COLOR_NAMES.split())


def render(
    shape,
    y_FOV,
    pose,
    scene,
    mask,
    lights,
    tiles,
    chunk_size,
    shadows=False,
    shadow_mask=None,
    num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    scene_args = scene, lights, mask, shadow_mask
    compiled = paz.graphics.scene.compile(*scene_args)
    args = shape, y_FOV, pose, compiled, tiles, chunk_size
    args = RenderArgs(*args, shadows, num_bounces, face_chunk_size)
    image, depth = scan_tiles(args, render_tile_step)
    return assemble_image(args, image), assemble_depth(args, depth)


def render_masks(
    shape,
    y_FOV,
    pose,
    scene,
    lights,
    depth,
    tiles,
    chunk_size,
    num_objects=None,
    shadows=False,
    shadow_mask=None,
    num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    if num_objects is None:
        num_objects = len(scene.nodes)
    min_depth, max_depth = depth
    num_nodes = len(scene.nodes)
    masks = []
    for object_arg in range(num_objects):
        mask = jp.zeros((num_nodes,), dtype=bool).at[object_arg].set(True)
        args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
        render_args = args + (shadows, shadow_mask, num_bounces)
        _, depth_image = render(*render_args, face_chunk_size)
        soft = paz.depth.to_soft_mask(depth_image, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def scan_tiles(args, render_step):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    paz.graphics.mesh.assert_exact_tile_side(H, H_tiles)
    paz.graphics.mesh.assert_exact_tile_side(W, W_tiles)
    coordinates = paz.graphics.mesh.make_tile_coordinates(H_tiles, W_tiles)
    render_step = paz.lock(render_step, args)
    return jax.lax.scan(render_step, None, coordinates)[1]


def render_tile_step(carry, tile_arg, args):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    camera_to_world = jp.linalg.inv(args.pose)
    tile_args = H, W, H_tiles, W_tiles, args.y_FOV, camera_to_world
    rays = paz.graphics.mesh.build_tile_rays(*tile_args, tile_arg)
    tile_H, tile_W = H // H_tiles, W // W_tiles
    trace_args = args.scene, args.shadows, args.num_bounces
    trace_args = trace_args + (args.face_chunk_size,)
    hit_mask, depth, color = trace_chunks(rays, trace_args, args.chunk_size)
    post_args = hit_mask, depth, color, args.pose, rays, tile_H, tile_W
    image, depth = postprocess(*post_args)
    return carry, (image, jp.expand_dims(depth, -1))


def assemble_image(args, image):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    return paz.graphics.mesh.assemble(H, W, H_tiles, W_tiles, image)


def assemble_depth(args, depth):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    return paz.graphics.mesh.assemble(H, W, H_tiles, W_tiles, depth)[..., 0]


def trace_chunks(rays, config, chunk_size):
    ray_chunks = split_ray_chunks(rays, chunk_size)
    trace_step = paz.lock(trace_chunk_step, config)
    hit_mask, depth, color = jax.lax.scan(trace_step, None, ray_chunks)[1]
    return flatten_chunk_results(hit_mask, depth, color, len(rays[0]))


def trace_chunk_step(carry, rays, config):
    compiled, shadows, num_bounces, face_chunk = config
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
        return array
    pad_size = chunk_size - remainder
    padding = jp.repeat(array[-1:], pad_size, axis=0)
    return jp.concatenate([array, padding], axis=0)


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
    args = color, depth, hit_mask, throughput, active_mask
    args += refractive_index, rays
    return RenderState(*args)


def bounce_step(state, bounce, compiled, shadows, face_chunk):
    triangle_hit = compute_triangle_hit(compiled, state.rays, face_chunk)
    intersections = intersect(compiled, state.rays, triangle_hit)
    hit_masks, depths, points, normals, indices, eyes = intersections
    hit_shape_args = find_closest_intersection_args(hit_masks, depths)
    closest = gather_closest(*intersections)
    state = update_first_hit(state, closest, bounce)
    state = update_active_mask(state, closest)
    args = state.rays, compiled, hit_shape_args, closest, points
    args += normals, eyes, shadows, triangle_hit
    colors = compute_hit_colors(*args)
    return update_state(state, compiled, closest, colors)


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


def update_first_hit(state, closest, bounce):
    if bounce != 0:
        return state
    return state._replace(depth=closest.depth, hit_mask=closest.hit_mask)


def update_active_mask(state, closest):
    active_mask = state.active_mask & closest.hit_mask
    return state._replace(active_mask=active_mask)


def compute_hit_colors(
    rays, compiled, indices, closest, points, normals, eyes, shadows,
    triangle_hit,
):
    shape_args = rays, compiled, indices, closest, points, normals, eyes
    num_shapes = len(compiled.shapes)
    if num_shapes == 0:
        colors = compute_triangle_colors(compiled, triangle_hit)
    elif triangle_hit is None:
        colors = compute_shape_colors(*shape_args, shadows)
    else:
        shape_colors = compute_shape_colors(*shape_args, shadows)
        triangle_colors = compute_triangle_colors(compiled, triangle_hit)
        is_triangle = jp.expand_dims(indices == num_shapes, -1)
        colors = jp.where(is_triangle, triangle_colors, shape_colors)
    return colors


def compute_shape_colors(
    rays, compiled, indices, closest, points, normals, eyes, shadows
):
    num_shapes = len(compiled.shapes)
    shape_args = jp.minimum(indices, num_shapes - 1)
    points, normals = points[:num_shapes], normals[:num_shapes]
    eyes = eyes[:num_shapes]
    if shadows:
        color_args = rays, compiled.shapes, compiled.lights, shape_args
        color_args += compiled.mask, compiled.shadow_mask
        color_args += closest.point, closest.normal, points, normals, eyes
        colors = color_with_shadows(ShadowColorArgs(*color_args))
    else:
        color_args = compiled.lights, compiled.shapes, points, normals
        colors = color_without_shadow(*color_args, eyes, shape_args)
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


def intersect(compiled, rays, triangle_hit):
    rows = []
    if len(compiled.shapes) > 0:
        rows.append(intersect_shapes(compiled.shapes, rays, compiled.mask))
    if triangle_hit is not None:
        rows.append(build_triangle_row(triangle_hit))
    return stack_intersection_rows(rows)


def intersect_shapes(shapes, rays, mask):

    def hide_shapes(mask, hit_masks):
        return jp.where(jp.expand_dims(mask, 1), hit_masks, False)

    merge = paz.graphics.shapes.field_merge(shapes, ["transform", "type"])
    intersect_fun = paz.lock(paz.graphics.shapes.intersect, *rays)
    hit_masks, depths, points, normals, eyes = jax.vmap(intersect_fun)(merge)
    return hide_shapes(mask, hit_masks), depths, points, normals, eyes


def build_triangle_row(triangle_hit):
    depth = jp.expand_dims(triangle_hit.depth, -1)
    rows = jp.expand_dims(triangle_hit.hit_mask, 0)
    rows = rows, jp.expand_dims(depth, 0)
    rows += (jp.expand_dims(triangle_hit.points, 0),)
    rows += (jp.expand_dims(triangle_hit.normals, 0),)
    return rows + (jp.expand_dims(triangle_hit.eyes, 0),)


def stack_intersection_rows(rows):
    joined = tuple(jp.concatenate(fields, axis=0) for fields in zip(*rows))
    indices = jp.arange(joined[0].shape[0])
    hit_masks, depths, points, normals, eyes = joined
    return hit_masks, depths, points, normals, indices, eyes


def gather_closest(hit_masks, depths, points, normals, indices, eyes):
    closest_args = find_closest_intersection_args(hit_masks, depths)
    args = take_closest(hit_masks, closest_args)
    args = args, take_closest(depths, closest_args)
    args += (take_closest(points, closest_args),)
    args += (take_closest(normals, closest_args),)
    args += (take_closest(eyes, closest_args),)
    args += (indices[closest_args],)
    return paz.graphics.Hit(*args)


def select_shader(material):
    if isinstance(material, paz.graphics.CookTorranceMaterial):
        return paz.graphics.cook_torrance
    return paz.graphics.phong


def color_without_shadow(lights, shapes, points, normals, eyes, hit_shape_args):
    colors, start_arg, merged_lights = [], 0, paz.graphics.shapes.merge(*lights)
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        final_arg = start_arg + len(group)
        group = paz.graphics.shapes.merge(*group)
        data = split_shape_data(points, normals, eyes, start_arg, final_arg)
        albedo = compute_group_albedo(group, data[0])
        args, axes = (albedo, group.material, *data), (0, 0, 0, 0, 0, None)
        shader = select_shader(group.material)
        color_per_light = jax.vmap(shader.compute_colors, axes)
        color = jax.vmap(color_per_light, (None, None, None, None, None, 0))
        colors.append(jp.sum(color(*args, merged_lights), axis=0))
        start_arg = final_arg
    return take_closest(jp.concatenate(colors, axis=0), hit_shape_args)


def compute_group_albedo(group, points):
    compute_albedo = paz.graphics.albedo.compute_shape_albedo
    return jax.vmap(compute_albedo)(group, group.material, points)


def split_shape_data(points, normals, eyes, start_arg, final_arg):
    points = points[start_arg:final_arg]
    normals = normals[start_arg:final_arg]
    eyes = eyes[start_arg:final_arg]
    return points, normals, eyes


def intersect_shape_groups(shapes, origins, directions, intersect_shape):

    def process_group(group, rays, start_arg):
        indices = jp.arange(start_arg, start_arg + len(group))
        merged_group = paz.graphics.shapes.merge(*group)
        intersect = paz.lock(intersect_shape, *rays)
        intersections = jax.vmap(intersect)(merged_group)
        return (*intersections, indices)

    def concatenate(x):
        return tuple(jp.concatenate(items, axis=0) for items in zip(*x))

    intersections, start_arg, rays = [], 0, (origins, directions)
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        intersections.append(process_group(group, rays, start_arg))
        start_arg = start_arg + len(group)
    return concatenate(intersections)


def intersect_shadow_groups(shapes, origins, directions):
    args = (shapes, origins, directions, paz.graphics.shapes.intersect_all)
    return intersect_shape_groups(*args)


def compute_surface_points(point, normal, epsilon=SHADOW_ORIGIN_EPSILON):
    over_point = point + normal * epsilon
    under_point = point - normal * epsilon
    return over_point, under_point


def compute_shadow_ray_origins(points, normals):
    over_point, _ = compute_surface_points(points, normals)
    return over_point


def compute_shadow_depth_thresholds(shape_indices, receiver_indices):
    same_shape = shape_indices[:, None] == receiver_indices[None, :]
    return jp.where(same_shape, SHADOW_SELF_HIT_EPSILON, paz.graphics.EPSILON)


def compute_front_side_shadow_mask(*args):
    shape_indices, receiver_indices, receiver_normals, directions = args
    same_shape = shape_indices[:, None] == receiver_indices[None, :]
    front_side = paz.algebra.dot(receiver_normals, directions) >= 0.0
    return jp.logical_and(same_shape, front_side[None, :])


def select_shadow_depths(*args):
    hit_masks, depths, shape_indices = args[:3]
    receiver_indices, receiver_normals, directions = args[3:]
    threshold_args = shape_indices, receiver_indices
    thresholds = compute_shadow_depth_thresholds(*threshold_args)
    front_args = shape_indices, receiver_indices, receiver_normals, directions
    front_side_hits = compute_front_side_shadow_mask(*front_args)
    valid_roots = depths > thresholds[:, None, :]
    valid_roots = jp.logical_and(valid_roots, depths < paz.graphics.FARAWAY)
    valid_roots = jp.logical_and(jp.expand_dims(hit_masks, 1), valid_roots)
    valid_roots = jp.logical_and(valid_roots, ~front_side_hits[:, None, :])
    depths = jp.where(valid_roots, depths, paz.graphics.FARAWAY)
    hit_masks = jp.any(valid_roots, axis=1)
    depths = jp.min(depths, axis=1)
    return hit_masks, depths


def compute_soft_occlusion(hit_masks, depths, light_lengths, slope=0.01):
    closest_depths = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_depths = jp.min(closest_depths, axis=0)
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    blockers = closest_depths <= light_lengths
    blocker_mask = jp.logical_and(scene_hit_mask, blockers)
    difference = light_lengths - closest_depths
    occlusion = jax.nn.sigmoid(slope * difference)
    return jp.where(blocker_mask, occlusion, 0.0)


def color_with_shadows(args):
    transparencies = compute_transparencies(args.shapes)
    colors = jp.zeros((len(args.points[0]), 3))
    lights = paz.graphics.shapes.merge(*args.lights)
    body = paz.lock(scan_light_step, args, transparencies)
    return jax.lax.scan(body, colors, lights)[0]


def scan_light_step(colors, light, args, transparencies):
    return colors + compute_light_colors(args, light, transparencies), None


def compute_transparencies(shapes):
    return jp.array([shape.material.transparency for shape in shapes])


def compute_light_colors(args, light, transparencies):
    directions, distance = compute_light_directions(light, args.point)
    origins = compute_shadow_ray_origins(args.point, args.normal)
    intersections = intersect_shadow_groups(args.shapes, origins, directions)
    hit_masks, depths, _, _, _, shape_indices = intersections
    masks = resolve_shadow_masks(args, hit_masks, transparencies)
    select_args = masks, depths, shape_indices, args.indices
    select_args += args.normal, directions
    masks, depths = select_shadow_depths(*select_args)
    is_shadow = compute_soft_occlusion(masks, depths, distance)
    color_args = args.shapes, light, args.points, args.normals, args.eyes
    color_args += (is_shadow,)
    colors = compute_shadowed_colors(*color_args)
    return take_closest(colors, args.indices)


def compute_light_directions(light, points):
    vector = light.position - points
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def resolve_shadow_masks(args, hit_masks, transparencies):
    shadow_masks = jp.where(jp.expand_dims(args.mask, 1), hit_masks, False)
    is_transparent = transparencies > 0.0
    shadow_masks = hide_transparent_shapes(shadow_masks, is_transparent)
    if args.shadow_mask is not None:
        cast_mask = jp.expand_dims(args.shadow_mask, 1)
        shadow_masks = jp.where(cast_mask, shadow_masks, False)
    return shadow_masks


def hide_transparent_shapes(shadow_masks, is_transparent):
    return jp.where(jp.expand_dims(is_transparent, 1), False, shadow_masks)


def compute_shadowed_colors(*args):
    shapes, light, points, normals, eyes, is_shadow = args
    colors, start_arg = [], 0
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        final_arg = start_arg + len(group)
        group = paz.graphics.shapes.merge(*group)
        data = split_shape_data(points, normals, eyes, start_arg, final_arg)
        albedo = compute_group_albedo(group, data[0])
        axes = 0, 0, 0, 0, 0, None, None
        shader = select_shader(group.material)
        color = jax.vmap(shader.compute_colors_with_shadow, axes)
        color_args = albedo, group.material, *data, light, is_shadow
        colors.append(color(*color_args))
        start_arg = final_arg
    return jp.concatenate(colors, axis=0)


def update_state(state, compiled, closest, intersected_colors):
    material = get_material_properties(compiled, closest.primitive_index)
    reflectivities, transparencies, refractivities = material
    color_args = state.color, state.throughput, state.active_mask
    color_args += intersected_colors, reflectivities, transparencies
    color = accumulate_color(*color_args)
    args = state.rays[1], state.refractive_index, closest.normal
    args += (refractivities,)
    normal, eye, n1, n2, n_ratio = prepare_computations(*args)
    reflectance = schlick(normal, eye, n1, n2)
    ray_args = normal, eye, n_ratio, closest.point, transparencies, reflectance
    new_rays = compute_new_rays(*ray_args)
    update_args = new_rays, n2, reflectivities, transparencies, reflectance
    return apply_bounce_update(state._replace(color=color), *update_args)


def get_material_properties(compiled, hit_shape_args):
    reflectivities, transparencies, refractivities = [], [], []
    for shape in compiled.shapes:
        reflectivities.append(shape.material.reflective)
        transparencies.append(shape.material.transparency)
        refractivities.append(shape.material.refractive_index)
    if compiled.triangles is not None:
        reflectivities.append(0.0)
        transparencies.append(0.0)
        refractivities.append(1.0)
    reflectivities = jp.array(reflectivities)[hit_shape_args]
    transparencies = jp.array(transparencies)[hit_shape_args]
    refractivities = jp.array(refractivities)[hit_shape_args]
    return reflectivities, transparencies, refractivities


def accumulate_color(*args):
    colors, throughput, active_mask = args[:3]
    intersected_colors, reflectivities, transparencies = args[3:]
    weights = jp.maximum(1.0 - reflectivities - transparencies, 0.0)
    weights = jp.expand_dims(weights, -1)
    active_mask = jp.expand_dims(active_mask, -1)
    return colors + (throughput * active_mask * weights * intersected_colors)


def flip_normal_if_inside(eye, normal):
    is_inside = jp.sum(normal * eye, axis=-1) < 0.0
    return jp.where(jp.expand_dims(is_inside, -1), -normal, normal), is_inside


def displace_by_normal(point, normal):
    upper_point = point + normal * BOUNCE_ORIGIN_EPSILON
    lower_point = point - normal * BOUNCE_ORIGIN_EPSILON
    return lower_point, upper_point


def prepare_computations(*args):
    current_directions, refractive_index, normal, refractive_indices = args
    eye = -current_directions
    normal, is_inside = flip_normal_if_inside(eye, normal)
    n1 = refractive_index
    n2 = jp.where(is_inside, 1.0, refractive_indices)  # TODO why 1.0 hardcoded
    n_ratio = n1 / (n2)
    return normal, eye, n1, n2, n_ratio


def schlick(normal, eye, n1, n2):
    n_ratio = n1 / n2
    cos_incident = jp.sum(eye * normal, axis=-1)
    sin_transmit_squared = (n_ratio**2) * (1.0 - (cos_incident**2))
    cos_transmit = jp.sqrt(jp.maximum(0.0, 1.0 - sin_transmit_squared))

    is_total_internal_reflection = sin_transmit_squared > 1.0
    cos = jp.where(n1 > n2, cos_transmit, cos_incident)

    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    reflectance = r0 + (1.0 - r0) * (1.0 - cos) ** 5
    return jp.where(is_total_internal_reflection, 1.0, reflectance)


def reflect_or_refract(transparancies, reflectance):
    is_transparent = transparancies > 0.0
    do_reflect = ~is_transparent
    return jp.expand_dims(do_reflect, -1)


def compute_refractive_direction(eye, normal, n_ratio):
    cos_incident = jp.sum(eye * normal, axis=-1)
    sin_transmit_squared = (n_ratio**2) * (1.0 - (cos_incident**2))
    cos_transmit = jp.sqrt(jp.maximum(0.0, 1.0 - sin_transmit_squared))
    inside_vector = -eye * jp.expand_dims(n_ratio, -1)
    up_weight = jp.expand_dims((n_ratio * cos_incident - cos_transmit), -1)
    return up_weight * normal + inside_vector


def compute_reflection_direction(eye, normal):
    return paz.graphics.geometry.reflect(-eye, normal)


def compute_new_rays(normal, eye, n_ratio, point, transparancies, reflectance):
    do_reflect = reflect_or_refract(transparancies, reflectance)
    reflection_direction = compute_reflection_direction(eye, normal)
    refractive_direction = compute_refractive_direction(eye, normal, n_ratio)
    direction = jp.where(do_reflect, reflection_direction, refractive_direction)
    direction = paz.algebra.normalize(direction)
    lower_point, upper_point = displace_by_normal(point, normal)
    origin = jp.where(do_reflect, upper_point, lower_point)
    return origin, direction


def apply_bounce_update(*args):
    state, new_rays, n2, reflectivities, transparencies, reflectance = args
    is_transparent = transparencies > 0.0
    is_reflective = reflectivities > 0.0
    factor_args = is_transparent, is_reflective, transparencies
    factor_args += reflectivities, reflectance
    factor = compute_bounce_factor(*factor_args)
    factor = jp.where(is_transparent & (reflectance >= 1.0), 1.0, factor)
    throughput = state.throughput * jp.expand_dims(factor, -1)
    active_mask = state.active_mask & (is_transparent | is_reflective)
    index_args = state, n2, is_transparent, reflectance
    refractive_index = update_refractive_index(*index_args)
    args = throughput, active_mask, refractive_index, new_rays
    return replace_bounce_state(state, *args)


def compute_bounce_factor(*args):
    is_transparent, is_reflective, transparencies = args[:3]
    reflectivities, reflectance = args[3:]
    transparent_factor = transparencies * (1.0 - reflectance)
    reflective_factor = jp.where(is_reflective, reflectivities, 0.0)
    return jp.where(is_transparent, transparent_factor, reflective_factor)


def update_refractive_index(state, n2, is_transparent, reflectance):
    update_mask = is_transparent & (reflectance < 1.0)
    return jp.where(update_mask, n2, state.refractive_index)


def replace_bounce_state(state, throughput, active_mask, index, rays):
    state = state._replace(throughput=throughput, active_mask=active_mask)
    return state._replace(refractive_index=index, rays=rays)
