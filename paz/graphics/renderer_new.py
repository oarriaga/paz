# TODO consider that the acne could be inside the renderer
# i.e. rays from scene to lights being tiny (floor hitting base).
import jax.numpy as jp
import paz
import jax


def render(image_shape, world_to_camera, rays, scene, lights, mask, shadows, shadow_mask=None, num_bounces=1):  # fmt: skip
    shapes, mask, shadow_mask, lights = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)  # fmt:skip
    args = (world_to_camera, rays, shapes, lights, mask, shadows, shadow_mask, num_bounces)  # fmt: skip
    return render_bounced(*image_shape, *args)


def render_bounced(H, W, world_to_camera, rays, shapes, lights, mask, shadows, shadow_mask, num_bounces):  # fmt: skip
    state = initialize_state(rays)
    bounce = paz.lock(bounce_step, shapes, lights, mask, shadows, shadow_mask)
    for step_arg in range(num_bounces):
        state = bounce(state, step_arg)
    hit_mask, depth, color = state["hit_mask"], state["depth"], state["color"]
    return postprocess(hit_mask, depth, color, world_to_camera, rays, H, W)


def initialize_state(rays):
    num_rays = rays[0].shape[0]
    return {
        "color": jp.zeros((num_rays, 3)),
        "depth": jp.full((num_rays,), paz.graphics.FARAWAY),
        "hit_mask": jp.zeros((num_rays,), dtype=bool),
        "throughput": jp.ones((num_rays, 3)),
        "active_mask": jp.ones((num_rays,), dtype=bool),
        "current_refractive_index": jp.ones((num_rays,)),
        "current_origins": rays[0],
        "current_directions": rays[1],
    }


def bounce_step(state, bounce, shapes, lights, mask, shadows, shadow_mask):
    rays = (state["current_origins"], state["current_directions"])
    intersections = intersect(shapes, rays, mask)
    hit_masks, depths, points, normals, indices, eyes = intersections
    hit_shape_args = find_closest_intersection_args(hit_masks, depths)
    closest = gather_closest(*intersections)
    if bounce == 0:
        state["depth"] = closest["depth"]
        state["hit_mask"] = closest["hit_mask"]
    state["active_mask"] &= closest["hit_mask"]

    if shadows:
        colors = color_with_shadows(rays, shapes, lights, hit_shape_args, mask, shadow_mask, closest["point"], points, normals, eyes)  # fmt: skip
    else:
        colors = color_without_shadow(lights, shapes, points, normals, eyes, hit_shape_args)  # fmt: skip
    return update_state(state, shapes, closest, colors)


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
    return {
        "hit_mask": take_closest(hit_masks, closest_args),
        "depth": take_closest(depths, closest_args),
        "point": take_closest(points, closest_args),
        "normal": take_closest(normals, closest_args),
        "eye": take_closest(eyes, closest_args),
        "shape_idx": indices[closest_args],
    }


def color_without_shadow(lights, shapes, points, normals, eyes, hit_shape_args):

    def split(points, normal, eyes, arg_0, arg_1):
        return points[arg_0:arg_1], normals[arg_0:arg_1], eyes[arg_0:arg_1]

    colors, start_arg, merged_lights = [], 0, paz.graphics.shapes.merge(*lights)
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        final_arg = start_arg + len(group)
        group = paz.graphics.shapes.merge(*group)
        data = split(points, normals, eyes, start_arg, final_arg)
        args, axes = (group, group.material, *data), (0, 0, 0, 0, 0, None)
        color_per_light = jax.vmap(paz.graphics.phong.compute_colors, axes)
        color = jax.vmap(color_per_light, (None, None, None, None, None, 0))
        colors.append(jp.sum(color(*args, merged_lights), axis=0))
        start_arg = final_arg
    return take_closest(jp.concatenate(colors, axis=0), hit_shape_args)


def intersect_groups(shapes, origins, directions):

    def process_group(group, rays, start_arg):
        indices = jp.arange(start_arg, start_arg + len(group))
        merged_group = paz.graphics.shapes.merge(*group)
        intersect = paz.lock(paz.graphics.shapes.intersect, *rays)
        intersections = jax.vmap(intersect)(merged_group)
        return (*intersections, indices)

    def concatenate(x):
        return tuple(jp.concatenate(items, axis=0) for items in zip(*x))

    intersections, start_arg, rays = [], 0, (origins, directions)
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        intersections.append(process_group(group, rays, start_arg))
        start_arg = start_arg + len(group)
    return concatenate(intersections)


def color_with_shadows(rays, shapes, lights, indices, mask, shadow_mask, closest_point, points, normals, eyes):  # fmt: skip
    transparencies = jp.array([shape.material.transparency for shape in shapes])

    def compute_light_colors(light):
        light_directions, distance = compute_light_directions(light, closest_point)  # fmt: skip
        intersections = intersect_groups(shapes, closest_point, light_directions)  # fmt:skip
        hit_masks, depths, _, _, _, _indices = intersections
        shadow_masks = resolve_shadow_masks(mask, shadow_mask, hit_masks, transparencies)  # fmt: skip
        is_shadow = calculate_occlusion(shadow_masks, depths, distance)
        colors = compute_shadowed_colors(shapes, light, points, normals, eyes, is_shadow)  # fmt: skip
        return take_closest(colors, indices)

    def compute_light_directions(light, points):
        vector = light.position - points
        norm = paz.algebra.compute_norms(vector, 1)
        return vector / norm, jp.squeeze(norm, axis=1)

    def resolve_shadow_masks(mask, shadow_mask, hit_masks, transparencies): # fmt: skip
        shadow_masks = jp.where(jp.expand_dims(mask, 1), hit_masks, False)  # fmt: skip
        is_transparent = transparencies > 0.0  # ignore if transparent
        shadow_masks = jp.where(jp.expand_dims(is_transparent, 1), False, shadow_masks) # fmt: skip
        if shadow_mask is not None:
            cast_mask = jp.expand_dims(shadow_mask, 1)
            shadow_masks = jp.where(cast_mask, shadow_masks, False)
        return shadow_masks

    def calculate_occlusion(masks, depths, light_length):
        scene_hit_mask = compute_scene_hit_mask(masks)
        depths = jp.where(masks, depths[..., 0], paz.graphics.FARAWAY)
        closest_depth = jp.min(depths, axis=0)
        is_shadow = jp.logical_and(scene_hit_mask, light_length > closest_depth)
        return is_shadow

    def compute_shadowed_colors(shapes, light, points, normals, eyes, is_shadow):  # fmt: skip

        def split(points, normal, eyes, arg_0, arg_1):
            return points[arg_0:arg_1], normals[arg_0:arg_1], eyes[arg_0:arg_1]

        colors, start_arg = [], 0
        for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
            final_arg = start_arg + len(group)
            group = paz.graphics.shapes.merge(*group)
            data = split(points, normals, eyes, start_arg, final_arg)
            args, axes = (group, group.material, *data), ( 0, 0, 0, 0, 0, None, None)  # fmt: skip
            color = jax.vmap(paz.graphics.phong.compute_colors_with_shadow, axes)  # fmt: skip
            colors.append(color(*args, light, is_shadow))
            start_arg = final_arg
        return jp.concatenate(colors, axis=0)

    colors = jp.zeros((len(points[0]), 3))
    for light in lights:
        colors = colors + compute_light_colors(light)
    return colors


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
    reflectivities, transparencies, refractivities = get_material_properties(shapes, closest["shape_idx"])  # fmt: skip
    state["color"] = accumulate_color(state["color"], state["throughput"], state["active_mask"], intersected_colors, reflectivities, transparencies)  # fmt:skip
    normalv, eyev, n1, n2, n_ratio, inside, lower_point, upper_point = _prepare_computations(state["current_directions"], state["current_refractive_index"], closest["point"], closest["normal"], refractivities)  # fmt: skip
    reflectance = schlick(normalv, eyev, n1, n1, n_ratio)
    new_rays = compute_new_rays(normalv, eyev, n_ratio, lower_point, upper_point, transparencies, reflectance)  # fmt: skip
    return _apply_bounce_update(state, *new_rays, n2, reflectivities, transparencies, reflectance)  # fmt: skip


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


def accumulate_color(colors, throughput, active_mask, intersected_colors, reflectivities, transparancies):  # fmt: skip
    weights = jp.maximum(1.0 - reflectivities - transparancies, 0.0)
    weights, active_mask = [jp.expand_dims(x, -1) for x in [weights, active_mask]]   # fmt: skip
    return colors + (throughput * active_mask * weights * intersected_colors)


def _prepare_computations(current_directions, now_refractive_index, point, normal, refractive_indices):   # fmt: skip
    eyev = -current_directions
    # Check if we are hitting the surface from the inside
    is_inside = jp.sum(normal * eyev, axis=-1) < 0.0
    # Flip normal if is_inside so it points against the ray
    normal = jp.where(jp.expand_dims(is_inside, -1), -normal, normal)

    n1 = now_refractive_index
    n2 = jp.where(is_inside, 1.0, refractive_indices)
    n_ratio = n1 / n2
    upper_point = point + normal * (paz.graphics.EPSILON * 1e-1)
    lower_point = point - normal * (paz.graphics.EPSILON * 1e-1)
    return normal, eyev, n1, n2, n_ratio, is_inside, lower_point, upper_point


def schlick(normalv, eyev, n1, n2, n_ratio):
    cos_i = jp.sum(eyev * normalv, axis=-1)

    sin2_t = (n_ratio**2) * (1.0 - cos_i**2)
    is_total_internal_reflection = sin2_t > 1.0

    cos_t = jp.sqrt(1.0 - sin2_t)
    cos = jp.where(n1 > n2, cos_t, cos_i)

    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    reflectance = r0 + (1.0 - r0) * (1.0 - cos) ** 5

    return jp.where(is_total_internal_reflection, 1.0, reflectance)


def reflect_or_refract(transparancies, reflectance):
    is_transparent = transparancies > 0.0
    is_total_internal_reflection = reflectance >= 1.0
    do_reflect = (~is_transparent) | is_total_internal_reflection
    return jp.expand_dims(do_reflect, -1)


def compute_reflection_direction(eyev, normalv):
    return paz.graphics.geometry.reflect(-eyev, normalv)


def compute_refraction_direction(eyev, normalv, n_ratio):
    cos_i = jp.sum(eyev * normalv, axis=-1)
    sin2_t = (n_ratio**2) * (1.0 - cos_i**2)
    cos_t = jp.sqrt(1.0 - sin2_t)
    n_ratio = jp.expand_dims(n_ratio, -1)
    return normalv * jp.expand_dims((n_ratio * cos_i - cos_t), -1) - eyev * n_ratio  # fmt: skip


def compute_new_rays(normalv, eyev, n_ratio, lower_point, upper_point, transparancies, reflectance):  # fmt: skip
    do_reflect = reflect_or_refract(transparancies, reflectance)
    refraction_direction = compute_refraction_direction(eyev, normalv, n_ratio)
    reflection_direction = compute_reflection_direction(eyev, normalv)
    direction = jp.where(do_reflect, reflection_direction, refraction_direction)
    origin = jp.where(do_reflect, upper_point, lower_point)
    return origin, direction


def _apply_bounce_update(state, next_origin, next_dir, n2, reflectivities, transparencies, reflectance):   # fmt: skip
    is_transparent = transparencies > 0.0
    is_reflective = reflectivities > 0.0
    factor = jp.where(is_transparent, transparencies * (1.0 - reflectance), jp.where(is_reflective, reflectivities, 0.0))  # fmt: skip
    factor = jp.where(is_transparent & (reflectance >= 1.0), 1.0, factor)
    state["throughput"] *= jp.expand_dims(factor, -1)
    state["active_mask"] &= is_transparent | is_reflective
    state["current_refractive_index"] = jp.where(is_transparent & (reflectance < 1.0), n2, state["current_refractive_index"])  # fmt: skip
    state["current_directions"] = next_dir
    state["current_origins"] = next_origin
    return state
