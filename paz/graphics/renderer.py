import jax.numpy as jp
import paz
import jax
from paz.graphics import EPSILON, FARAWAY


def postprocess(
    hit_masks, depths, colors, world_to_camera, rays, height, width
):
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    scene_colors = select_colors(depths, colors)
    image = to_color_image(scene_hit_mask, scene_colors, height, width)
    depth = to_depth_image(
        scene_hit_mask, depths, world_to_camera, rays, height, width
    )
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


def intersect_groups(shapes, origins, directions):
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    shape_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}

    results = [
        process_shape_group(group, origins, directions, shape_to_arg)
        for group in grouped_shapes.values()
    ]

    return concatenate_intersections(results)


def process_shape_group(group, origins, directions, shape_to_arg):
    group_indices = jp.array([shape_to_arg[id(shape)] for shape in group])
    merged_group = paz.graphics.shapes.merge(*group)
    intersect_fn = paz.lock(paz.graphics.shapes.intersect, origins, directions)
    vmap_intersect = jax.vmap(intersect_fn)

    intersections = vmap_intersect(merged_group)
    # Unpack: hit_masks, depths, points, normals, eyes
    return (*intersections, group_indices)


def concatenate_intersections(results):
    # results is list of (hit_mask, depth, point, normal, eye, index) tuples
    if not results:
        # TODO: Handle empty case or assume shapes always exist?
        return [], [], [], [], [], []

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
    for image_size, shape_group in grouped_shapes.items():
        shape_group = paz.graphics.shapes.merge(*shape_group)
        fn = jax.vmap(paz.lock(render_shape, lights, rays, shadow_mask))
        group_hit_masks, group_depths, group_colors = fn(shape_group)
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
    colors = compute_scene_colors(
        shape, lights, points, normals, eyes, shadow_mask
    )
    return hit_mask, depth, colors


def compute_scene_colors(shape, lights, points, normals, eyes, shadow_mask):
    color_args = (shape, shape.material, points, normals, eyes)
    color = paz.partial(
        paz.graphics.phong.compute_colors_with_shadow, *color_args
    )
    scene_colors = jp.array([color(light, shadow_mask) for light in lights])
    scene_colors = jp.sum(scene_colors, axis=0)
    return scene_colors


def compute_occlusion(
    points_to_light_norms, points_to_light_depth, points_to_light_hit_mask
):
    first_hit_light_source = points_to_light_norms > points_to_light_depth
    is_shadow = jp.logical_and(points_to_light_hit_mask, first_hit_light_source)
    return is_shadow


def compute_soft_occlusion(
    points_to_light_norms,
    points_to_light_depth,
    points_to_light_hit_mask,
    slope=0.01,
):
    occlusion_value = points_to_light_depth - points_to_light_norms
    occlusion_factor = jax.nn.sigmoid(-slope * occlusion_value)
    occlusion_factor = jp.where(points_to_light_hit_mask, occlusion_factor, 0.0)
    return occlusion_factor


def take_closest(array, indices):
    # array: (T, N, ...)
    # indices: (N,)
    # Target indices shape: (1, N, 1, ..., 1) matching array.ndim

    ndim = array.ndim
    # indices is 1D (N,).
    # We want (1, N, 1, ..., 1).

    # First expand 0 (T dim)
    idx = jp.expand_dims(indices, 0)  # (1, N)

    # Then expand tail
    while idx.ndim < ndim:
        idx = jp.expand_dims(idx, -1)

    val = jp.take_along_axis(array, idx, axis=0)
    return jp.squeeze(val, axis=0)


def _initialize_render_state(rays):
    num_rays = rays[0].shape[0]
    initial_state = {
        "accumulated_color": jp.zeros((num_rays, 3)),
        "accumulated_depth": jp.full((num_rays,), FARAWAY),
        "accumulated_hit_mask": jp.zeros((num_rays,), dtype=bool),
        "throughput": jp.ones((num_rays, 3)),
        "active_mask": jp.ones((num_rays,), dtype=bool),
        "current_ior": jp.ones((num_rays,)),  # Index of Refraction (Air)
        "current_origins": rays[0],
        "current_directions": rays[1],
    }
    return initial_state


def _compute_bounce_intersections(shapes, rays, mask):
    hit_masks, depths, points, normals, eyes, indices = intersect_groups(
        shapes, *rays
    )
    bounce_mask = mask[indices]
    bounce_mask = jp.expand_dims(bounce_mask, 1)
    hit_masks = jp.where(bounce_mask, hit_masks, False)
    return hit_masks, depths, points, normals, eyes, indices


def _find_closest_intersection(hit_masks, depths):
    depths_masked = jp.where(jp.expand_dims(hit_masks, -1), depths, FARAWAY)
    closest_indices = jp.argmin(depths_masked, axis=0)
    closest_indices = jp.squeeze(closest_indices, -1)
    return closest_indices


def _gather_closest_attributes(
    closest_indices, hit_masks, depths, points, normals, indices
):
    closest = {
        "hit_mask": take_closest(hit_masks, closest_indices),
        "depth": take_closest(depths, closest_indices),
        "point": take_closest(points, closest_indices),
        "normal": take_closest(normals, closest_indices),
        "shape_idx": indices[closest_indices],
    }
    return closest


def _compute_lighting(
    state,
    shapes,
    lights,
    closest_indices,
    closest_point,
    shadows,
    mask,
    shadow_mask_init,
):
    if shadows:
        num_rays = state["current_origins"].shape[0]
        return _compute_shadow_occlusion_loop(
            shapes,
            lights,
            mask,
            shadow_mask_init,
            closest_point,
            num_rays,
            state,
            closest_indices,
        )
    return _compute_simple_lighting(state, shapes, lights, closest_indices)


def _compute_simple_lighting(state, shapes, lights, indices):
    num_rays = state["current_origins"].shape[0]
    dummy_shadow = jp.zeros((num_rays,), dtype=float)
    _, _, colors = render_shapes(
        shapes,
        lights,
        (state["current_origins"], state["current_directions"]),
        dummy_shadow,
    )
    return take_closest(colors, indices)


def _compute_shadow_occlusion_loop(
    shapes,
    lights,
    mask,
    shadow_init,
    point,
    num_rays,
    state,
    indices,
):
    local_colors = jp.zeros((num_rays, 3))
    for light in lights:
        contribution = _compute_single_light_contribution(
            light, shapes, mask, shadow_init, point, state, indices
        )
        local_colors += contribution
    return local_colors


def _compute_single_light_contribution(
    light, shapes, mask, shadow_init, point, state, indices
):
    direction, distance = _compute_light_vectors(light, point)
    hits = intersect_groups(shapes, point, direction)
    shadow_masks = _resolve_shadow_masks(mask, shadow_init, hits)
    occlusion = _calculate_occlusion(shadow_masks, hits[1], distance)
    _, _, colors = render_shapes(
        shapes,
        [light],
        (state["current_origins"], state["current_directions"]),
        occlusion,
    )
    return take_closest(colors, indices)


def _compute_light_vectors(light, point):
    vector = light.position - point
    norm = paz.algebra.compute_norms(vector, 1)
    return vector / norm, jp.squeeze(norm, axis=1)


def _resolve_shadow_masks(mask, shadow_init, hits):
    sh_masks, indices = hits[0], hits[5]
    sh_masks = jp.where(jp.expand_dims(mask[indices], 1), sh_masks, False)
    if shadow_init is not None:
        cast_mask = jp.expand_dims(shadow_init[indices], 1)
        sh_masks = jp.where(cast_mask, sh_masks, False)
    return sh_masks


def _calculate_occlusion(masks, depths, light_dist):
    scene_mask = compute_scene_hit_mask(masks)
    depths = jp.where(masks, depths[..., 0], FARAWAY)
    closest_depth = jp.min(depths, axis=0)
    return compute_soft_occlusion(light_dist, closest_depth, scene_mask)


def _process_bounce(
    state, shapes, lights, mask, shadows, shadow_mask_init, bounce
):
    intersections = _compute_bounce_intersections(
        shapes, (state["current_origins"], state["current_directions"]), mask
    )
    hit_masks, depths, points, normals, eyes, indices = intersections
    closest_indices = _find_closest_intersection(hit_masks, depths)
    closest = _gather_closest_attributes(
        closest_indices, hit_masks, depths, points, normals, indices
    )

    if bounce == 0:
        state["accumulated_depth"] = closest["depth"]
        state["accumulated_hit_mask"] = closest["hit_mask"]

    state["active_mask"] &= closest["hit_mask"]
    local_colors = _compute_lighting(
        state, shapes, lights, closest, closest_indices, shadows, mask, shadow_mask_init
    )
    return _update_bounce_state(state, shapes, closest, local_colors)


def _update_bounce_state(state, shapes, closest, local_color):
    materials = _extract_material_properties(shapes, closest["shape_idx"])
    _accumulate_local_color(state, local_color, materials)

    actions = _determine_bounce_actions(materials)
    next_dir, next_ior = _calculate_next_bounce_vectors(
        state, closest["normal"], materials, actions
    )

    return _apply_bounce_update(
        state, closest["point"], next_dir, next_ior, materials, actions
    )


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

    state["accumulated_color"] += (
        state["throughput"]
        * contribution
        * jp.expand_dims(state["active_mask"], -1)
    )


def _determine_bounce_actions(materials):
    is_refractive = materials["refractive"] > 0.0
    is_reflective = materials["reflective"] > 0.0

    do_refract = is_refractive
    do_reflect = is_reflective & (~do_refract)
    return do_refract, do_reflect


def _calculate_next_bounce_vectors(state, normal, materials, actions):
    do_refract, do_reflect = actions
    incident = state["current_directions"]
    current_ior = state["current_ior"]

    is_entering = jp.abs(current_ior - 1.0) < 1e-3
    target_ior = jp.where(is_entering, materials["ior"], 1.0)

    refract_dir = paz.graphics.geometry.refract(
        incident, normal, current_ior, target_ior
    )
    reflect_dir = paz.graphics.geometry.reflect(incident, normal)

    next_dir = jp.where(
        jp.expand_dims(do_refract, -1),
        refract_dir,
        jp.where(jp.expand_dims(do_reflect, -1), reflect_dir, incident),
    )
    next_ior = jp.where(do_refract, target_ior, current_ior)

    return next_dir, next_ior


def _apply_bounce_update(state, point, next_dir, next_ior, materials, actions):
    do_refract, do_reflect = actions
    
    factor = jp.where(
        do_refract,
        materials["refractive"],
        jp.where(do_reflect, materials["reflective"], 0.0),
    )
    state["throughput"] *= jp.expand_dims(factor, -1)

    state["active_mask"] &= (do_refract | do_reflect)
    state["current_ior"] = next_ior
    state["current_directions"] = next_dir
    state["current_origins"] = point + next_dir * EPSILON

    return state


def _compute_lighting(
    state,
    shapes,
    lights,
    closest,
    closest_indices,
    shadows,
    mask,
    shadow_mask_init,
):
    num_rays = state["current_origins"].shape[0]
    local_colors = jp.zeros((num_rays, 3))

    if shadows:
        local_colors = _compute_shadow_occlusion_loop(
            shapes,
            lights,
            mask,
            shadow_mask_init,
            closest["point"],
            num_rays,
            state,
            closest_indices,
        )
    else:
        dummy_shadow = jp.zeros((num_rays,), dtype=float)
        _, _, all_colors = render_shapes(
            shapes,
            lights,
            (state["current_origins"], state["current_directions"]),
            dummy_shadow,
        )
        local_colors = take_closest(all_colors, closest_indices)
    return local_colors


def _compute_shadow_occlusion_loop(
    shapes,
    lights,
    mask,
    shadow_mask_init,
    closest_point,
    num_rays,
    state,
    closest_indices,
):
    local_colors = jp.zeros((num_rays, 3))
    for light in lights:
        points_to_light = light.position - closest_point
        points_to_light_norms = paz.algebra.compute_norms(points_to_light, 1)
        points_to_light = points_to_light / points_to_light_norms

        sh_hits = intersect_groups(shapes, closest_point, points_to_light)
        sh_masks, sh_depths, _, _, _, sh_indices = sh_hits

        sh_obj_mask = mask[sh_indices]
        sh_masks = jp.where(
            jp.expand_dims(sh_obj_mask, 1), sh_masks, False
        )

        if shadow_mask_init is not None:
            sh_cast_mask = shadow_mask_init[sh_indices]
            sh_masks = jp.where(
                jp.expand_dims(sh_cast_mask, 1), sh_masks, False
            )

        sh_hit_mask = compute_scene_hit_mask(sh_masks)
        sh_depths = jp.where(sh_masks, sh_depths[..., 0], FARAWAY)
        sh_depth = jp.min(sh_depths, axis=0)
        points_to_light_norms = jp.squeeze(points_to_light_norms, axis=1)

        occlusion = compute_soft_occlusion(
            points_to_light_norms, sh_depth, sh_hit_mask
        )

        _, _, l_colors = render_shapes(
            shapes,
            [light],
            (state["current_origins"], state["current_directions"]),
            occlusion,
        )
        local_colors += take_closest(l_colors, closest_indices)
    return local_colors


def _postprocess_render(state, world_to_camera, rays, image_shape):
    final_hit_mask = jp.expand_dims(state["accumulated_hit_mask"], 0)
    final_depths = jp.expand_dims(state["accumulated_depth"], 0)
    final_colors = jp.expand_dims(state["accumulated_color"], 0)
    return postprocess(
        final_hit_mask,
        final_depths,
        final_colors,
        world_to_camera,
        rays,
        *image_shape,
    )


def _render_bounced(
    image_shape,
    world_to_camera,
    rays,
    shapes,
    lights,
    mask,
    shadows,
    shadow_mask_init,
    max_bounces=3,
):
    state = _initialize_render_state(rays)
    for bounce in range(max_bounces):
        state = _process_bounce(
            state, shapes, lights, mask, shadows, shadow_mask_init, bounce
        )
    return _postprocess_render(
        state, world_to_camera, rays, image_shape
    )


def render(
    image_shape,
    world_to_camera,
    rays,
    scene,
    lights,
    mask,
    shadows,
    shadow_mask=None,
):
    shapes, lights, mask = paz.graphics.scene.compile(scene, lights, mask)

    if shadows and shadow_mask is not None:
        shadow_mask = paz.graphics.scene.prepare_mask(
            shadow_mask, len(shapes), scene
        )
    else:
        shadow_mask = None

    renderer = _render_bounced(
        image_shape,
        world_to_camera,
        rays,
        shapes,
        lights,
        mask,
        shadows,
        shadow_mask,
    )
    return renderer


def _render(image_shape, world_to_camera, rays, shapes, lights, mask):
    return _render_bounced(
        image_shape,
        world_to_camera,
        rays,
        shapes,
        lights,
        mask,
        shadows=False,
        shadow_mask_init=None,
    )


def _render_with_shadows(
    image_shape, world_to_camera, rays, shapes, lights, mask, shadow_mask
):
    return _render_bounced(
        image_shape,
        world_to_camera,
        rays,
        shapes,
        lights,
        mask,
        shadows=True,
        shadow_mask_init=shadow_mask,
    )
