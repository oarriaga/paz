from collections import namedtuple

import jax
import jax.numpy as jp

import paz

SHADOW_ORIGIN_EPSILON = 1e-3
SHADOW_SELF_HIT_EPSILON = 1e-3
BOUNCE_ORIGIN_EPSILON = 1e-3
RENDER_NAMES = "shape y_FOV pose scene mask lights tiles chunk_size shadows "
RENDER_NAMES += "shadow_mask num_bounces"
STATE_NAMES = "color depth hit_mask throughput active_mask "
STATE_NAMES += "refractive_index rays"
CLOSEST_NAMES = "hit_mask depth point normal eye shape_idx"
SHADOW_COLOR_NAMES = "rays shapes lights indices mask shadow_mask "
SHADOW_COLOR_NAMES += "point normal points normals eyes"

RenderArgs = namedtuple("RenderArgs", RENDER_NAMES.split())
RenderState = namedtuple("RenderState", STATE_NAMES.split())
ClosestHit = namedtuple("ClosestHit", CLOSEST_NAMES.split())
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
):
    args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
    args += shadows, shadow_mask, num_bounces
    args = RenderArgs(*args)
    args = compile_render_args(args)
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
):
    if num_objects is None:
        num_objects = len(scene.nodes)
    min_depth, max_depth = depth
    num_nodes = len(scene.nodes)
    masks = []
    for object_arg in range(num_objects):
        mask = jp.zeros((num_nodes,), dtype=bool).at[object_arg].set(True)
        args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
        _, depth_image = render(*args, shadows, shadow_mask, num_bounces)
        soft = paz.depth.to_soft_mask(depth_image, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def compile_render_args(args):
    scene_args = args.scene, args.lights, args.mask, args.shadow_mask
    shapes, mask, shadow_mask, lights = paz.graphics.scene.compile(*scene_args)
    return args._replace(
        scene=shapes,
        mask=mask,
        shadow_mask=shadow_mask,
        lights=lights,
    )


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
    trace_args = args.scene, args.lights, args.mask, args.shadows
    trace_args += (args.shadow_mask,)
    trace_args += (args.num_bounces,)
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
    shapes, lights, mask, shadows, shadow_mask, num_bounces = config
    args = rays, shapes, lights, mask, shadows, shadow_mask, num_bounces
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


def trace_bounces(rays, shapes, lights, mask, shadows, shadow_mask, bounces):
    state = initialize_state(rays)
    bounce = paz.lock(bounce_step, shapes, lights, mask, shadows, shadow_mask)
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


def bounce_step(state, bounce, shapes, lights, mask, shadows, shadow_mask):
    intersections = intersect(shapes, state.rays, mask)
    hit_masks, depths, points, normals, indices, eyes = intersections
    hit_shape_args = find_closest_intersection_args(hit_masks, depths)
    closest = gather_closest(*intersections)
    state = update_first_hit(state, closest, bounce)
    state = update_active_mask(state, closest)
    args = state.rays, shapes, lights, hit_shape_args, mask, shadow_mask
    args += closest, points, normals, eyes, shadows
    colors = compute_hit_colors(*args)
    return update_state(state, shapes, closest, colors)


def update_first_hit(state, closest, bounce):
    if bounce != 0:
        return state
    return state._replace(depth=closest.depth, hit_mask=closest.hit_mask)


def update_active_mask(state, closest):
    active_mask = state.active_mask & closest.hit_mask
    return state._replace(active_mask=active_mask)


def compute_hit_colors(*args):
    rays, shapes, lights, indices, mask, shadow_mask = args[:6]
    closest, points, normals, eyes, shadows = args[6:]
    if shadows:
        color_args = rays, shapes, lights, indices, mask, shadow_mask
        color_args += closest.point, closest.normal, points, normals, eyes
        return color_with_shadows(ShadowColorArgs(*color_args))
    args = lights, shapes, points, normals, eyes, indices
    return color_without_shadow(*args)


def intersect(shapes, rays, mask):

    def hide_shapes(mask, hit_masks):
        return jp.where(jp.expand_dims(mask, 1), hit_masks, False)

    merge = paz.graphics.shapes.field_merge(shapes, ["transform", "type"])
    indices = jp.arange(len(shapes))
    intersect_fun = paz.lock(paz.graphics.shapes.intersect, *rays)
    hit_masks, depths, points, normals, eyes = jax.vmap(intersect_fun)(merge)
    hit_masks = hide_shapes(mask, hit_masks)
    return hit_masks, depths, points, normals, indices, eyes


def find_closest_intersection_args(hit_masks, depths):
    if depths.ndim > hit_masks.ndim:
        depths = jp.squeeze(depths, axis=-1)
    depths_masked = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_args = jp.argmin(depths_masked, axis=0)
    return closest_args


def take_closest(candidates, closest_indices):
    num_feature_dims = candidates.ndim - 2
    broadcast_shape = (1, -1) + (1,) * num_feature_dims
    indices = closest_indices.reshape(broadcast_shape)
    selected = jp.take_along_axis(candidates, indices, axis=0)
    return jp.squeeze(selected, axis=0)


def gather_closest(hit_masks, depths, points, normals, indices, eyes):
    closest_args = find_closest_intersection_args(hit_masks, depths)
    args = take_closest(hit_masks, closest_args)
    args = args, take_closest(depths, closest_args)
    args += (take_closest(points, closest_args),)
    args += (take_closest(normals, closest_args),)
    args += (take_closest(eyes, closest_args),)
    args += (indices[closest_args],)
    return ClosestHit(*args)


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
        args, axes = (group, group.material, *data), (0, 0, 0, 0, 0, None)
        shader = select_shader(group.material)
        color_per_light = jax.vmap(shader.compute_colors, axes)
        color = jax.vmap(color_per_light, (None, None, None, None, None, 0))
        colors.append(jp.sum(color(*args, merged_lights), axis=0))
        start_arg = final_arg
    return take_closest(jp.concatenate(colors, axis=0), hit_shape_args)


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
        axes = 0, 0, 0, 0, 0, None, None
        shader = select_shader(group.material)
        color = jax.vmap(shader.compute_colors_with_shadow, axes)
        color_args = group, group.material, *data, light, is_shadow
        colors.append(color(*color_args))
        start_arg = final_arg
    return jp.concatenate(colors, axis=0)


def postprocess(hit_masks, depths, colors, world_to_camera, rays, H, W):
    hit_masks = jp.expand_dims(hit_masks, 0)
    depths = jp.expand_dims(depths, 0)
    colors = jp.expand_dims(colors, 0)
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    scene_colors = select_colors(depths, colors)
    image = to_color_image(scene_hit_mask, scene_colors, H, W)
    depth = to_depth_image(scene_hit_mask, depths, world_to_camera, rays, H, W)
    return image, depth


def compute_scene_hit_mask(hit_masks):
    hit_masks = jp.array(hit_masks)
    hit_mask = jp.sum(hit_masks, axis=0)
    hit_mask = hit_mask.astype(bool)
    return hit_mask


def select_colors(depths, colors):
    depths = jp.array(depths)
    arg_depths = jp.argmin(depths, axis=0)
    arg_depths = jp.expand_dims(arg_depths, 0)

    colors = jp.array(colors)
    ndim = colors.ndim
    while arg_depths.ndim < ndim:
        arg_depths = jp.expand_dims(arg_depths, -1)

    colors = jp.take_along_axis(colors, arg_depths, axis=0)
    colors = jp.squeeze(colors, axis=0)
    return colors


def to_color_image(hit_mask, colors, height, width, background_color=1):
    image = jp.where(hit_mask, colors.T, background_color)
    image = image.reshape((3, height, width))
    image = jp.clip(image[:], 0, 1)
    image = jp.rollaxis(image, 0, 3)
    return image


def to_depth_image(hit_mask, depths, world_to_camera, rays, height, width):
    min_depths = jp.min(jp.array(depths), axis=0)
    world_depths = calculate_world_depth(min_depths, world_to_camera, rays)
    masked_depths = apply_hit_mask_to_depth(hit_mask, world_depths)
    return reshape_depth_image(masked_depths, height, width)


def calculate_world_depth(min_depths, world_to_camera, rays):
    points = paz.graphics.geometry.compute_points3D(*rays, min_depths)
    points = paz.algebra.transform_points(world_to_camera, points)
    return -points[:, -1]


def apply_hit_mask_to_depth(hit_mask, depths, faraway=0):
    return jp.where(hit_mask, depths, faraway)


def reshape_depth_image(depths, height, width):
    image = depths.reshape((1, height, width))
    image = jp.rollaxis(image, 0, 3)
    return image[:, :, 0]


def update_state(state, shapes, closest, intersected_colors):
    material = get_material_properties(shapes, closest.shape_idx)
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


def get_material_properties(shapes, hit_shape_args):
    reflectivities, transparencies, refractivities = [], [], []
    for shape in shapes:
        reflectivities.append(shape.material.reflective)
        transparencies.append(shape.material.transparency)
        refractivities.append(shape.material.refractive_index)
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
