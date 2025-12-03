import jax.numpy as jp
import paz
import jax


def postprocess(hit_masks, depths, colors, world_to_camera, rays, H, W):
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


def to_color_image(hit_mask, colors, H, W, background_color=1):
    image = jp.where(hit_mask, colors.T, background_color)
    image = image.reshape((3, H, W))
    image = jp.clip(image[:], 0, 1)
    image = jp.rollaxis(image, 0, 3)
    return image


def to_depth_image(hit_mask, depths, world_to_camera, rays, H, W):
    min_depths = jp.min(jp.array(depths), axis=0)
    world_depths = calculate_world_depth(min_depths, world_to_camera, rays)
    masked_depths = apply_hit_mask_to_depth(hit_mask, world_depths)
    return reshape_depth_image(masked_depths, H, W)


def calculate_world_depth(min_depths, world_to_camera, rays):
    points = paz.graphics.geometry.compute_points3D(*rays, min_depths)
    points = paz.algebra.transform_points(world_to_camera, points)
    return -points[:, -1]


def apply_hit_mask_to_depth(hit_mask, depths, faraway=0):
    return jp.where(hit_mask, depths, faraway)


def reshape_depth_image(depths, H, W):
    image = depths.reshape((1, H, W))
    image = jp.rollaxis(image, 0, 3)
    return image[:, :, 0]


def intersect_groups(shapes, origins, directions):
    groups = paz.graphics.shapes.group_by_pattern_size(shapes)
    shape_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    args = (origins, directions, shape_to_arg)
    intersections = [process_group(group, *args) for group in groups.values()]
    return concatenate(intersections)


def process_group(group, origins, directions, shape_to_arg):
    group_indices = jp.array([shape_to_arg[id(shape)] for shape in group])
    merged_group = paz.graphics.shapes.merge(*group)
    intersect = paz.lock(paz.graphics.shapes.intersect, origins, directions)
    intersections = jax.vmap(intersect)(merged_group)
    return (*intersections, group_indices)


def concatenate(results):
    return tuple(jp.concatenate(items, axis=0) for items in zip(*results))


def get_closest_hit_points(depths, points):
    min_depth_indices = jp.argmin(depths, axis=0)
    min_depth_indices = jp.expand_dims(min_depth_indices, axis=0)
    closest_points = jp.take_along_axis(points, min_depth_indices, axis=0)
    closest_points = jp.squeeze(closest_points, axis=0)
    return closest_points


def render_shapes(shapes, lights, rays, shadow_mask):
    hit_masks, depths, colors = [], [], []
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    render_shapes = jax.vmap(paz.lock(render_shape, lights, rays, shadow_mask))
    for image_size, shape_group in grouped_shapes.items():
        shape_group = paz.graphics.shapes.merge(*shape_group)
        group_hit_masks, group_depths, group_colors = render_shapes(shape_group)
        hit_masks.append(group_hit_masks)
        depths.append(group_depths)
        colors.append(group_colors)
    hit_masks = jp.concatenate(hit_masks, axis=0)
    depths = jp.concatenate(depths, axis=0)
    colors = jp.concatenate(colors, axis=0)
    return hit_masks, depths, colors


def render_shape(shape, lights, rays, shadow_mask):
    intersections = paz.graphics.shapes.intersect(shape, *rays)
    hit_mask, depth, points, normals, eyes = intersections
    args = (shape, lights, points, normals, eyes, shadow_mask)
    colors = compute_scene_colors(*args)
    return hit_mask, depth, colors


def compute_scene_colors(shape, lights, points, normals, eyes, shadow_mask):
    args = (shape, shape.material, points, normals, eyes)
    color = paz.partial(paz.graphics.phong.compute_colors_with_shadow, *args)
    scene_colors = jp.array([color(light, shadow_mask) for light in lights])
    scene_colors = jp.sum(scene_colors, axis=0)
    return scene_colors


def _calculate_occlusion(masks, depths, light_length):
    scene_mask = compute_scene_hit_mask(masks)
    depths = jp.where(masks, depths[..., 0], paz.graphics.FARAWAY)
    closest_depth = jp.min(depths, axis=0)
    return compute_soft_occlusion(light_length, closest_depth, scene_mask, 0.01)


def compute_occlusion(light_length, closest_depth, scene_hit_mask):
    depths_lower_than_light_distance = light_length > closest_depth
    is_shadow = jp.logical_and(scene_hit_mask, depths_lower_than_light_distance)
    return is_shadow


def compute_soft_occlusion(light_length, closest_depth, scene_hit_mask, slope):
    occlusion_value = closest_depth - light_length
    occlusion_factor = jax.nn.sigmoid(-slope * occlusion_value)
    occlusion_factor = jp.where(scene_hit_mask, occlusion_factor, 0.0)
    return occlusion_factor


def _compute_bounce_intersections(shapes, rays, mask):
    intersections = intersect_groups(shapes, *rays)
    hit_masks, depths, points, normals, eyes, indices = intersections
    bounce_mask = jp.expand_dims(mask[indices], 1)
    hit_masks = jp.where(bounce_mask, hit_masks, False)
    return hit_masks, depths, points, normals, eyes, indices


def take_closest(candidates, closest_indices):
    # candidates (num_shapes, num_rays, *features), closest_indices (num_rays,)
    # reshape indices to (1, num_rays, 1, ..., 1) for broadcasting
    num_feature_dims = candidates.ndim - 2
    broadcast_shape = (1, -1) + (1,) * num_feature_dims
    indices = closest_indices.reshape(broadcast_shape)
    # select closest candidate per ray
    selected = jp.take_along_axis(candidates, indices, axis=0)
    return jp.squeeze(selected, axis=0)


def _initialize_render_state(rays):
    num_rays = rays[0].shape[0]
    return {
        "accumulated_color": jp.zeros((num_rays, 3)),
        "accumulated_depth": jp.full((num_rays,), paz.graphics.FARAWAY),
        "accumulated_hit_mask": jp.zeros((num_rays,), dtype=bool),
        "throughput": jp.ones((num_rays, 3)),
        "active_mask": jp.ones((num_rays,), dtype=bool),
        "current_IoR": jp.ones((num_rays,)),  # Index of Refraction (Air)
        "current_origins": rays[0],
        "current_directions": rays[1],
    }


def _gather_closest(closest_args, hit_masks, depths, points, normals, shapes):
    return {
        "hit_mask": take_closest(hit_masks, closest_args),
        "depth": take_closest(depths, closest_args),
        "point": take_closest(points, closest_args),
        "normal": take_closest(normals, closest_args),
        "shape_idx": shapes[closest_args],
    }


def _compute_simple_lighting(rays, shapes, lights, indices):
    dummy_shadow = jp.zeros((rays[0].shape[0],), dtype=float)
    _, _, colors = render_shapes(shapes, lights, rays, dummy_shadow)
    return take_closest(colors, indices)


def _resolve_shadow_masks(mask, shadow_mask, intersections):
    sh_masks, indices = intersections[0], intersections[5]
    sh_masks = jp.where(jp.expand_dims(mask[indices], 1), sh_masks, False)
    if shadow_mask is not None:
        cast_mask = jp.expand_dims(shadow_mask[indices], 1)
        sh_masks = jp.where(cast_mask, sh_masks, False)
    return sh_masks


def _compute_light_colors(light, shapes, mask, shadow_mask, point, rays, indices):  # fmt: skip
    direction, distance = _compute_light_vectors(light, point)
    intersections = intersect_groups(shapes, point, direction)
    shadow_masks = _resolve_shadow_masks(mask, shadow_mask, intersections)
    occlusion = _calculate_occlusion(shadow_masks, intersections[1], distance)
    _, _, colors = render_shapes(shapes, [light], rays, occlusion)
    return take_closest(colors, indices)


def _compute_light_vectors(light, point):
    vector = light.position - point
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def _compute_shadows(rays, shapes, lights, indices, mask, shadow_mask, points):
    colors = jp.zeros((len(rays[0]), 3))
    args = (shapes, mask, shadow_mask, points, rays, indices)
    for light in lights:
        colors = colors + _compute_light_colors(light, *args)
    return colors


def _compute_lighting(rays, shapes, lights, closest_args, closest_points, shadows, mask, shadow_mask):   # fmt: skip
    args = (rays, shapes, lights, closest_args)
    if shadows:
        colors = _compute_shadows(*args, mask, shadow_mask, closest_points)
    else:
        colors = _compute_simple_lighting(*args)
    return colors


def _find_closest_intersection(hit_masks, depths):
    # Ensure depths and hit_masks are broadcast compatible
    # If depths is (T, N, 1) and hit_masks is (T, N), squeeze depths
    if depths.ndim > hit_masks.ndim:
        depths = jp.squeeze(depths, axis=-1)
    depths_masked = jp.where(hit_masks, depths, paz.graphics.FARAWAY)
    closest_args = jp.argmin(depths_masked, axis=0)
    return closest_args


def _bounce(state, bounce, shapes, lights, mask, shadows, shadow_mask):
    rays = (state["current_origins"], state["current_directions"])
    intersections = _compute_bounce_intersections(shapes, rays, mask)
    hit_masks, depths, points, normals, eyes, indices = intersections
    closest_args = _find_closest_intersection(hit_masks, depths)
    closest = _gather_closest(closest_args, hit_masks, depths, points, normals, indices)  # fmt: skip

    if bounce == 0:
        state["accumulated_depth"] = closest["depth"]
        state["accumulated_hit_mask"] = closest["hit_mask"]

    state["active_mask"] &= closest["hit_mask"]
    colors = _compute_lighting(rays, shapes, lights, closest_args, closest["point"], shadows, mask, shadow_mask) # fmt: skip
    return _update_bounce_state(state, shapes, closest, colors)


def _update_bounce_state(state, shapes, closest, local_color):
    materials = _extract_material_properties(shapes, closest["shape_idx"])
    state = _accumulate_local_color(state, local_color, materials)
    actions = _determine_bounce_actions(materials)
    next_dir, next_ior = _calculate_next_bounce_vectors(state, closest["normal"], materials, actions)  # fmt: skip
    return _apply_bounce_update(state, closest["point"], next_dir, next_ior, materials, actions)  # fmt: skip


def _extract_material_properties(shapes, shape_indices):
    reflective = jp.array([s.material.reflective for s in shapes])
    refractive = jp.array([s.material.refractive for s in shapes])
    ior = jp.array([s.material.refractive_index for s in shapes])
    return {
        "reflective": reflective[shape_indices],
        "refractive": refractive[shape_indices],
        "ior": ior[shape_indices],
    }


def _accumulate_local_color(state, local_color, materials):
    weight = 1.0 - materials["reflective"] - materials["refractive"]
    weight = jp.maximum(weight, 0.0)
    contribution = local_color * jp.expand_dims(weight, -1)
    state["accumulated_color"] += ( state["throughput"] * contribution * jp.expand_dims(state["active_mask"], -1))  # fmt: skip
    return state


def _determine_bounce_actions(materials):
    is_refractive = materials["refractive"] > 0.0
    is_reflective = materials["reflective"] > 0.0
    do_refract = is_refractive
    do_reflect = is_reflective & (~do_refract)
    return do_refract, do_reflect


def _calculate_next_bounce_vectors(state, normal, materials, actions):
    do_refract, do_reflect = actions
    incident = state["current_directions"]
    current_IoR = state["current_IoR"]

    is_entering = jp.abs(current_IoR - 1.0) < 1e-3
    target_ior = jp.where(is_entering, materials["ior"], 1.0)

    refract_dir = paz.graphics.geometry.refract(
        incident, normal, current_IoR, target_ior
    )
    reflect_dir = paz.graphics.geometry.reflect(incident, normal)

    next_dir = jp.where(
        jp.expand_dims(do_refract, -1),
        refract_dir,
        jp.where(jp.expand_dims(do_reflect, -1), reflect_dir, incident),
    )
    next_ior = jp.where(do_refract, target_ior, current_IoR)

    return next_dir, next_ior


def _apply_bounce_update(state, point, next_dir, next_ior, materials, actions):
    do_refract, do_reflect = actions
    factor = jp.where(
        do_refract,
        materials["refractive"],
        jp.where(do_reflect, materials["reflective"], 0.0),
    )
    state["throughput"] *= jp.expand_dims(factor, -1)
    state["active_mask"] &= do_refract | do_reflect
    state["current_IoR"] = next_ior
    state["current_directions"] = next_dir
    state["current_origins"] = point + next_dir * paz.graphics.EPSILON
    return state


def _postprocess_render(state, world_to_camera, rays, H, W):
    hit_mask = jp.expand_dims(state["accumulated_hit_mask"], 0)
    depths = jp.expand_dims(state["accumulated_depth"], 0)
    colors = jp.expand_dims(state["accumulated_color"], 0)
    return postprocess(hit_mask, depths, colors, world_to_camera, rays, H, W)


def _render_bounced(H, W, world_to_camera, rays, shapes, lights, mask, shadows, shadow_mask, max_bounces=3):  # fmt: skip
    state = _initialize_render_state(rays)
    bounce = paz.lock(_bounce, shapes, lights, mask, shadows, shadow_mask)
    for step_arg in range(max_bounces):
        state = bounce(state, step_arg)
    return _postprocess_render(state, world_to_camera, rays, H, W)


def render(image_shape, world_to_camera, rays, scene, lights, mask, shadows, shadow_mask=None):  # fmt: skip
    shapes, lights, mask = paz.graphics.scene.compile(scene, lights, mask)
    if shadows and shadow_mask is not None:
        shadow_mask = paz.graphics.scene.prepare_mask(shadow_mask, len(shapes), scene)  # fmt: skip
    args = (world_to_camera, rays, shapes, lights, mask, shadows, shadow_mask)
    return _render_bounced(*image_shape, *args)
