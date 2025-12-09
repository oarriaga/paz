# TODO consider that the acne could be inside the renderer
# i.e. rays from scene to lights being tiny (floor hitting base).
import jax.numpy as jp
import paz
import jax
from paz.graphics.types import Shape, Material, Pattern


def render(image_shape, world_to_camera, rays, scene, lights, mask, shadows, shadow_mask=None, num_bounces=1):  # fmt: skip
    shapes, lights, mask = paz.graphics.scene.compile(scene, lights, mask)
    if shadows and shadow_mask is not None:
        shadow_mask_args = (shadow_mask, len(shapes), scene)
        shadow_mask = paz.graphics.scene.prepare_mask(*shadow_mask_args)
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
    closest_args = find_closest_intersection_args(hit_masks, depths)
    closest = gather_closest(*intersections)
    if bounce == 0:
        state["depth"] = closest["depth"]
        state["hit_mask"] = closest["hit_mask"]
    state["active_mask"] &= closest["hit_mask"]

    if shadows:
        colors = color_with_shadows(rays, shapes, lights, closest_args, mask, shadow_mask, closest["point"], points, normals, eyes)  # fmt: skip
    else:
        colors = color_without_shadows(lights, shapes, closest["point"], closest["normal"], closest["eye"], closest["shape_idx"])   # fmt: skip
    return update_state(state, shapes, closest, colors)


def hide_shapes(mask, hit_masks, shape_order):
    mask = jp.expand_dims(mask[shape_order], 1)
    return jp.where(mask, hit_masks, False)


def intersect(shapes, rays, mask):
    intersections = intersect_groups(shapes, *rays)
    hit_masks, depths, points, normals, eyes, indices = intersections
    hit_masks = hide_shapes(mask, hit_masks, indices)
    return hit_masks, depths, points, normals, indices, eyes


def intersect_groups(shapes, origins, directions):

    def process_group(group, rays, ID_to_arg):
        indices = jp.array([ID_to_arg[id(shape)] for shape in group])
        # shape_order = indices
        merged_group = paz.graphics.shapes.merge(*group)
        intersect = paz.lock(paz.graphics.shapes.intersect, *rays)
        intersections = jax.vmap(intersect)(merged_group)
        return (*intersections, indices)

    def concatenate(x):
        return tuple(jp.concatenate(items, axis=0) for items in zip(*x))

    groups = paz.graphics.shapes.group_by_pattern_size(shapes)
    ID_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    args = ((origins, directions), ID_to_arg)
    intersections = [process_group(group, *args) for group in groups.values()]
    return concatenate(intersections)


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


def color_without_shadows(lights, shapes, points, normals, eyes, shape_order):

    def expand_last(*args):
        return [jp.expand_dims(arg, 1) for arg in args]

    def batch(group, args, subtype):
        Type = {"material": Material, "pattern": Pattern}[subtype]
        type_data = {}
        for field in Type._fields:
            data = [getattr(getattr(shape, subtype), field) for shape in group]
            type_data[field] = jp.array(data)
        return Type(**{field: data[args] for field, data in type_data.items()})

    def gather(group, args):
        material = batch(group, args, "material")
        patterns = batch(group, args, "pattern")
        transform = jp.array([shape.transform for shape in group])[args]
        shapetype = jp.array([shape.type for shape in group])[args]
        return Shape(transform, shapetype, material, patterns)

    def build_local_map(shapes, group_indices):
        local_map = jp.zeros(len(shapes), int)
        return local_map.at[group_indices].set(jp.arange(len(group_indices)))

    def compute(group, ID_to_arg):
        group_indices = jp.array([ID_to_arg[id(shape)] for shape in group])
        local_map = build_local_map(shapes, group_indices)
        args_to_gather = local_map[shape_order]
        shape = gather(group, args_to_gather)
        args = (shape, shape.material, *expand_last(points, normals, eyes))
        axes = (0, 0, 0, 0, 0, None)
        color = jax.vmap(paz.graphics.phong.compute_colors, axes)

        mask = jp.any(shape_order[:, None] == group_indices[None, :], 1, keepdims=True)  # fmt: skip

        return sum(jp.squeeze(color(*args, light), 1) for light in lights), mask

    ID_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    colors = jp.zeros((len(points), 3))
    for group in paz.graphics.shapes.group_by_pattern_size(shapes).values():
        group_colors, mask = compute(group, ID_to_arg)
        colors = jp.where(mask, group_colors, colors)
    return colors


def color_with_shadows(rays, shapes, lights, indices, mask, shadow_mask, closest_point, points, normals, eyes):  # fmt: skip
    transparencies = jp.array([shape.material.transparency for shape in shapes])

    def compute_light_colors(light):
        light_directions, distance = compute_light_directions(light, closest_point)  # fmt: skip
        intersections = intersect_groups(shapes, closest_point, light_directions)  # fmt:skip
        hit_masks, depths, _, _, _, _indices = intersections
        shadow_masks = resolve_shadow_masks(mask, shadow_mask, hit_masks, _indices, transparencies)  # fmt: skip
        is_shadow = calculate_occlusion(shadow_masks, depths, distance)
        colors = compute_shadowed_colors(shapes, light, points, normals, eyes, is_shadow)  # fmt: skip
        return take_closest(colors, indices)

    def compute_light_directions(light, points):
        vector = light.position - points
        norm = paz.algebra.compute_norms(vector, 1)
        return vector / norm, jp.squeeze(norm, axis=1)

    def resolve_shadow_masks(mask, shadow_mask, hit_masks, indices, transparencies): # fmt: skip
        shadow_masks = jp.where(jp.expand_dims(mask[indices], 1), hit_masks, False)  # fmt: skip
        is_transparent = transparencies[indices] > 0.0  # ignore if transparent
        shadow_masks = jp.where(jp.expand_dims(is_transparent, 1), False, shadow_masks) # fmt: skip
        if shadow_mask is not None:
            cast_mask = jp.expand_dims(shadow_mask[indices], 1)
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


def update_state(state, shapes, closest, local_color):
    materials = get_material_properties(shapes, closest["shape_idx"])
    state = accumulate_local_color(state, local_color, materials)
    computations = _prepare_computations(state, closest, materials)
    reflectance = schlick(computations)
    next_dir, next_origin = determine_next_bounce(computations, materials, reflectance)  # fmt: skip
    return _apply_bounce_update(state, next_origin, next_dir, computations, materials, reflectance)  # fmt: skip


def get_material_properties(shapes, shape_args):
    reflective = jp.array([shape.material.reflective for shape in shapes])
    transparency = jp.array([shape.material.transparency for shape in shapes])
    refractive_index = jp.array([shape.material.refractive_index for shape in shapes])  # fmt: skip
    return {
        "reflective": reflective[shape_args],
        "transparency": transparency[shape_args],
        "refractive_index": refractive_index[shape_args],
    }


def accumulate_local_color(state, local_color, materials):
    weight = 1.0 - materials["reflective"] - materials["transparency"]
    weight = jp.maximum(weight, 0.0)
    contribution = local_color * jp.expand_dims(weight, -1)
    state["color"] += (
        state["throughput"]
        * contribution
        * jp.expand_dims(state["active_mask"], -1)
    )
    return state


def _prepare_computations(state, closest, materials):
    eyev = -state["current_directions"]
    normalv = closest["normal"]

    # Check if we are hitting the surface from the inside
    dot = jp.sum(normalv * eyev, axis=-1)
    inside = dot < 0.0

    # Flip normal if inside so it points against the ray
    normalv = jp.where(jp.expand_dims(inside, -1), -normalv, normalv)

    n1 = state["current_refractive_index"]

    # FIX:
    # If entering (inside=False), n2 is material's index.
    # If exiting (inside=True), n2 is 1.0 (Air).
    n2 = jp.where(inside, 1.0, materials["refractive_index"])

    n_ratio = n1 / n2

    # over_point = closest["point"] + normalv * (paz.graphics.EPSILON / 2.0)
    # under_point = closest["point"] - normalv * (paz.graphics.EPSILON / 2.0)
    over_point = closest["point"] + normalv * (paz.graphics.EPSILON)
    under_point = closest["point"] - normalv * (paz.graphics.EPSILON)

    return {
        "eyev": eyev,
        "normalv": normalv,
        "inside": inside,
        "n1": n1,
        "n2": n2,
        "n_ratio": n_ratio,
        "over_point": over_point,
        "under_point": under_point,
    }


def schlick(computations):
    cos_i = jp.sum(computations["eyev"] * computations["normalv"], axis=-1)

    sin2_t = (computations["n_ratio"] ** 2) * (1.0 - cos_i**2)
    is_total_internal_reflection = sin2_t > 1.0

    cos_t = jp.sqrt(1.0 - sin2_t)
    cos = jp.where(computations["n1"] > computations["n2"], cos_t, cos_i)

    r0 = (
        (computations["n1"] - computations["n2"])
        / (computations["n1"] + computations["n2"])
    ) ** 2
    reflectance = r0 + (1.0 - r0) * (1.0 - cos) ** 5

    return jp.where(is_total_internal_reflection, 1.0, reflectance)


def determine_next_bounce(computations, materials, reflectance):
    is_transparent = materials["transparency"] > 0.0
    is_total_internal_reflection = reflectance >= 1.0

    should_reflect = (~is_transparent) | is_total_internal_reflection

    reflect_dir = paz.graphics.geometry.reflect(
        -computations["eyev"], computations["normalv"]
    )

    cos_i = jp.sum(computations["eyev"] * computations["normalv"], axis=-1)
    sin2_t = (computations["n_ratio"] ** 2) * (1.0 - cos_i**2)
    cos_t = jp.sqrt(1.0 - sin2_t)

    direction = computations["normalv"] * jp.expand_dims(
        (computations["n_ratio"] * cos_i - cos_t), -1
    ) - computations["eyev"] * jp.expand_dims(computations["n_ratio"], -1)

    refract_dir = direction

    next_dir = jp.where(
        jp.expand_dims(should_reflect, -1), reflect_dir, refract_dir
    )
    next_origin = jp.where(
        jp.expand_dims(should_reflect, -1),
        computations["over_point"],
        computations["under_point"],
    )

    return next_dir, next_origin


def _apply_bounce_update(
    state, next_origin, next_dir, computations, materials, reflectance
):
    is_transparent = materials["transparency"] > 0.0
    is_reflective = materials["reflective"] > 0.0

    factor = jp.where(
        is_transparent,
        materials["transparency"] * (1.0 - reflectance),
        jp.where(is_reflective, materials["reflective"], 0.0),
    )

    factor = jp.where(is_transparent & (reflectance >= 1.0), 1.0, factor)

    state["throughput"] *= jp.expand_dims(factor, -1)
    state["active_mask"] &= is_transparent | is_reflective
    state["current_refractive_index"] = jp.where(
        is_transparent & (reflectance < 1.0),
        computations["n2"],
        state["current_refractive_index"],
    )
    state["current_directions"] = next_dir
    state["current_origins"] = next_origin

    return state


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
