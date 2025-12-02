import jax.numpy as jp
import paz
import jax
from paz.graphics import EPSILON, FARAWAY


def render_shapes(shapes, lights, rays):
    shape_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    hit_masks, depths, colors, indices = [], [], [], []
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    for image_size, shape_group in grouped_shapes.items():
        group_indices = [shape_to_arg[id(shape)] for shape in shape_group]
        indices.append(jp.array(group_indices))
        shape_group = paz.graphics.shapes.merge(*shape_group)
        vmap_render = jax.vmap(paz.lock(render_shape, lights, rays))
        group_hit_masks, group_depths, group_colors = vmap_render(shape_group)
        hit_masks.append(group_hit_masks)
        depths.append(group_depths)
        colors.append(group_colors)
    hit_masks = jp.concatenate(hit_masks, axis=0)
    depths = jp.concatenate(depths, axis=0)
    colors = jp.concatenate(colors, axis=0)
    indices = jp.concatenate(indices, axis=0)
    return hit_masks, depths, colors, indices


def render_shape(shape, lights, rays):
    intersections = paz.graphics.shapes.intersect(shape, *rays)
    hit_mask, depth, points, normals, eyes = intersections
    colors = compute_scene_colors(shape, lights, points, normals, eyes)
    return hit_mask, depth, colors


def compute_scene_colors(shape, lights, points, normals, eyes):
    color_args = (shape, shape.material, points, normals, eyes)
    color = paz.partial(paz.graphics.phong.compute_colors, *color_args)
    scene_colors = jp.array([color(light) for light in lights])
    scene_colors = jp.sum(scene_colors, axis=0)
    return scene_colors


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
    colors = jp.take_along_axis(colors, arg_depths, axis=0)
    colors = jp.squeeze(colors, axis=0)
    return colors


def to_color_image(hit_mask, colors, H, W, background_color=1):
    image = jp.where(hit_mask, colors.T, background_color)
    image = image.reshape((3, H, W))
    image = jp.clip(image[:], 0, 1)
    image = jp.rollaxis(image, 0, 3)
    return image


def to_depth_image(hit_mask, depths, world_to_camera, rays, H, W, faraway=0):
    depths = jp.array(depths)
    depth_image = jp.min(depths, axis=0)
    depth_image = paz.graphics.geometry.compute_points3D(*rays, depth_image)
    depth_image = paz.algebra.transform_points(world_to_camera, depth_image)
    depth_image = -depth_image[:, -1]
    depth_image = jp.where(hit_mask, depth_image, faraway)
    depth_image = depth_image.reshape((1, H, W))
    depth_image = jp.rollaxis(depth_image, 0, 3)
    depth_image = depth_image[:, :, 0]
    return depth_image


def intersect_groups(shapes, origins, directions):
    shape_to_arg = {id(shape): arg for arg, shape in enumerate(shapes)}
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    hit_masks, depths, points, normals, eyes, indices = [], [], [], [], [], []
    intersect = paz.lock(paz.graphics.shapes.intersect, origins, directions)
    for image_size, group in grouped_shapes.items():
        group_indices = [shape_to_arg[id(shape)] for shape in group]
        indices.append(jp.array(group_indices))
        group = paz.graphics.shapes.merge(*group)
        intersections = jax.vmap(intersect)(group)
        hit_masks.append(intersections[0])
        depths.append(intersections[1])
        points.append(intersections[2])
        normals.append(intersections[3])
        eyes.append(intersections[4])
    hit_masks = jp.concatenate(hit_masks, axis=0)
    depths = jp.concatenate(depths, axis=0)
    points = jp.concatenate(points, axis=0)
    normals = jp.concatenate(normals, axis=0)
    eyes = jp.concatenate(eyes, axis=0)
    indices = jp.concatenate(indices, axis=0)
    return hit_masks, depths, points, normals, eyes, indices


def get_closest_hit_points(depths, points):
    min_depth_indices = jp.argmin(depths, axis=0)
    min_depth_indices = jp.expand_dims(min_depth_indices, axis=0)
    closest_points = jp.take_along_axis(points, min_depth_indices, axis=0)
    closest_points = jp.squeeze(closest_points, axis=0)
    return closest_points


def _render_shapes(shapes, lights, rays, shadow_mask):
    hit_masks, depths, colors = [], [], []
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    for image_size, shape_group in grouped_shapes.items():
        shape_group = paz.graphics.shapes.merge(*shape_group)
        fn = jax.vmap(paz.lock(_render_shape, lights, rays, shadow_mask))
        group_hit_masks, group_depths, group_colors = fn(shape_group)
        hit_masks.append(group_hit_masks)
        depths.append(group_depths)
        colors.append(group_colors)
    hit_masks = jp.concatenate(hit_masks, axis=0)
    depths = jp.concatenate(depths, axis=0)
    colors = jp.concatenate(colors, axis=0)
    return hit_masks, depths, colors


def _render_shape(shape, lights, rays, shadow_mask):
    intersections = paz.graphics.shapes.intersect(shape, *rays)
    hit_mask, depth, points, normals, eyes = intersections
    colors = _compute_scene_colors(
        shape, lights, points, normals, eyes, shadow_mask
    )
    return hit_mask, depth, colors


def _compute_scene_colors(shape, lights, points, normals, eyes, shadow_mask):
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
    num_rays = rays[0].shape[0]
    H, W = image_shape

    # Extract material properties
    mat_reflective = jp.array([s.material.reflective for s in shapes])
    mat_refractive = jp.array([s.material.refractive for s in shapes])
    mat_refractive_index = jp.array(
        [s.material.refractive_index for s in shapes]
    )
    # mat_color = jp.array([s.material.color for s in shapes])

    accumulated_color = jp.zeros((num_rays, 3))
    accumulated_depth = jp.full((num_rays,), FARAWAY)
    accumulated_hit_mask = jp.zeros((num_rays,), dtype=bool)
    throughput = jp.ones((num_rays, 3))
    active_mask = jp.ones((num_rays,), dtype=bool)
    current_ior = jp.ones((num_rays,))  # Air

    current_origins, current_directions = rays

    for bounce in range(max_bounces):
        # Intersect geometry
        hit_masks, depths, points, normals, eyes, indices = intersect_groups(
            shapes, current_origins, current_directions
        )

        # Apply mask
        bounce_mask = mask[indices]
        bounce_mask = jp.expand_dims(bounce_mask, 1)
        hit_masks = jp.where(bounce_mask, hit_masks, False)

        # Find closest
        # depths is (T, N, 1), hit_masks is (T, N). Expand hit_masks.
        depths_masked = jp.where(jp.expand_dims(hit_masks, -1), depths, FARAWAY)
        closest_indices = jp.argmin(depths_masked, axis=0)
        # argmin over (T, N, 1) -> (N, 1). Squeeze it.
        closest_indices = jp.squeeze(closest_indices, -1)

        closest_hit_mask = take_closest(hit_masks, closest_indices)
        closest_depth = take_closest(depths, closest_indices)
        closest_point = take_closest(points, closest_indices)
        closest_normal = take_closest(normals, closest_indices)
        # closest_eye = take_closest(eyes, closest_indices)
        closest_shape_idx = indices[closest_indices]

        # Update Active Mask
        active_mask = active_mask & closest_hit_mask

        # For the first bounce, save depth and hit mask for output
        if bounce == 0:
            accumulated_depth = closest_depth
            accumulated_hit_mask = closest_hit_mask

        # Shadow Calculation
        # We construct a shadow mask for _render_shapes.
        # Ideally we compute shadows only for closest points.
        # To use _render_shapes, we need a shadow mask of shape (TotalHits, NumRays).
        # We can compute it for all, or just sparse.
        # Let's compute for closest and scatter.
        is_shadow_sparse = jp.zeros((num_rays,), dtype=bool)

        if shadows:
            # Move points slightly to avoid self-intersection
            shadow_origins = paz.pointcloud.move_along_normals(
                closest_point, closest_normal, EPSILON
            )

            # We check shadows per light... actually _compute_scene_colors iterates lights.
            # _render_with_shadows logic:
            # It accumulates colors manually iterating lights.
            # _render_shapes assumes a single shadow mask? No, _compute_scene_colors calls `color(light, shadow_mask)`.
            # And `_compute_scene_colors` sums over lights.
            # Wait, `_compute_scene_colors` takes `shadow_mask`.
            # But `shadow_mask` usually is per-light?
            # In `_render_with_shadows`, it iterates lights:
            #   is_shadow = compute_soft_occlusion(...)
            #   _render_shapes(..., is_shadow)
            # So _render_shapes is called PER LIGHT?
            # Yes! `color_shapes.append(colors)`.
            # So to support shadows correctly, we must loop over lights here too.
            pass

        # To keep it simple and compatible with existing shadows logic:
        # If shadows are enabled, we loop lights. If not, we call _render_shapes once.

        local_colors = jp.zeros((num_rays, 3))

        if shadows:
            for light in lights:
                points_to_light = light.position - closest_point
                points_to_light_norms = paz.algebra.compute_norms(
                    points_to_light, 1
                )
                points_to_light = points_to_light / points_to_light_norms

                # Cast shadow ray
                shadow_hits = intersect_groups(
                    shapes, closest_point, points_to_light
                )[:2]
                sh_masks, sh_depths = shadow_hits

                # Apply mask to shadow hits too?
                # Mask usually applies to visibility.
                # sh_indices = shadow_hits[5] # intersect_groups returns 6 items.
                # But I sliced [:2].
                # We should probably respect mask for shadows too.
                # For simplicity/speed, assuming masked objects don't cast shadows.
                # Re-calling intersect_groups fully to get indices:
                sh_masks, sh_depths, _, _, _, sh_indices = intersect_groups(
                    shapes, closest_point, points_to_light
                )
                sh_obj_mask = mask[sh_indices]
                sh_masks = jp.where(
                    jp.expand_dims(sh_obj_mask, 1), sh_masks, False
                )

                # User provided shadow_mask (init)
                if shadow_mask_init is not None:
                    # shadow_mask_init corresponds to shapes?
                    # In _render_with_shadows:
                    # shadow_mask = shadow_mask[indices]
                    # points_to_light_hit_masks = where(shadow_mask, ...)
                    # This implies shadow_mask_init masks objects that CAST shadows?
                    # Yes.
                    sh_cast_mask = shadow_mask_init[sh_indices]
                    sh_masks = jp.where(
                        jp.expand_dims(sh_cast_mask, 1), sh_masks, False
                    )

                sh_hit_mask = compute_scene_hit_mask(sh_masks)

                sh_depths = jp.where(sh_masks, sh_depths[..., 0], FARAWAY)
                sh_depth = jp.min(sh_depths, axis=0)
                # sh_depth = jp.squeeze(sh_depth, axis=1) # already (NumRays,)
                points_to_light_norms = jp.squeeze(
                    points_to_light_norms, axis=1
                )

                occlusion = compute_soft_occlusion(
                    points_to_light_norms, sh_depth, sh_hit_mask
                )

                # Now we need to shade.
                # We can't easily use _render_shapes per light because we need to scatter occlusion?
                # No, we can compute color just for the closest point!
                # We don't need _render_shapes for all points.
                # We have `closest_point`, `closest_normal`, `closest_shape_idx`.
                # But `closest_shape_idx` varies per ray.
                # We can't vmap `compute_colors` over different materials easily if we don't have a "Material" array struct.
                # BUT `_render_shapes` DOES exactly that by grouping!
                # So we should use `_render_shapes`.
                # We pass the occlusion (NumRays,) directly. It will be applied to ALL shapes for that ray.
                # Since we only pick the closest shape's color, and occlusion is for that shape, it works.

                # Now call _render_shapes for this light
                _, _, l_colors = _render_shapes(
                    shapes,
                    [light],
                    (current_origins, current_directions),
                    occlusion,
                )

                # Accumulate
                l_closest_color = take_closest(l_colors, closest_indices)
                local_colors = local_colors + l_closest_color
        else:
            # No shadows
            # Pass dummy shadow mask (zeros)
            dummy_shadow = jp.zeros((num_rays,), dtype=float)
            _, _, all_colors = _render_shapes(
                shapes,
                lights,
                (current_origins, current_directions),
                dummy_shadow,
            )
            local_colors = take_closest(all_colors, closest_indices)

        # Mix Material
        c_reflective = mat_reflective[closest_shape_idx]
        c_refractive = mat_refractive[closest_shape_idx]
        c_ior = mat_refractive_index[closest_shape_idx]

        # Local Contribution
        # Scale by (1 - refl - refr)
        weight = 1.0 - c_reflective - c_refractive
        weight = jp.maximum(weight, 0.0)

        contribution = local_colors * jp.expand_dims(weight, -1)

        # Add to accumulator (masked by active)
        accumulated_color = accumulated_color + (
            throughput * contribution * jp.expand_dims(active_mask, -1)
        )

        # Next Bounce Setup
        is_refractive = c_refractive > 0.0
        is_reflective = c_reflective > 0.0

        do_refract = is_refractive
        do_reflect = is_reflective & (~do_refract)

        incident = current_directions
        normal = closest_normal

        # IOR
        # If current_ior is close to 1.0, we are entering?
        # Or check if we are hitting the same material?
        # Simplification: If current == 1.0, entering -> n2=c_ior. Else exiting -> n2=1.0.
        is_entering = jp.abs(current_ior - 1.0) < 1e-3
        n2 = jp.where(is_entering, c_ior, 1.0)

        refract_dir = paz.graphics.geometry.refract(
            incident, normal, current_ior, n2
        )
        reflect_dir = paz.graphics.geometry.reflect(incident, normal)

        next_dir = jp.where(
            jp.expand_dims(do_refract, -1),
            refract_dir,
            jp.where(jp.expand_dims(do_reflect, -1), reflect_dir, incident),
        )

        # Update Throughput
        # factor = where(refract, refractive, where(reflect, reflective, 0))
        factor = jp.where(
            do_refract, c_refractive, jp.where(do_reflect, c_reflective, 0.0)
        )
        throughput = throughput * jp.expand_dims(factor, -1)

        # Stop if factor is 0 (neither reflected nor refracted)
        continues = do_refract | do_reflect
        active_mask = active_mask & continues

        # Update IOR
        current_ior = jp.where(do_refract, n2, current_ior)

        # Update Rays
        current_directions = next_dir
        current_origins = closest_point + current_directions * EPSILON

    # Postprocess
    # We need "hit_masks" (for postprocess to know background).
    # Use accumulated_hit_mask (from first bounce).
    # depths -> accumulated_depth.

    # hit_masks expects (1, NumRays) usually?
    # _render returns postprocess args.
    # postprocess expects: hit_masks, depths, colors ...
    # In original code: hit_masks is (TotalHits, NumRays).
    # But postprocess calls compute_scene_hit_mask which sums them.
    # If we pass a single hit mask (NumRays,), we should reshape to (1, NumRays).

    final_hit_mask = jp.expand_dims(accumulated_hit_mask, 0)
    final_depths = jp.expand_dims(accumulated_depth, 0)
    final_colors = jp.expand_dims(accumulated_color, 0)
    # select_colors expects (Total, Rays, 3). argmin(depths).
    # If we pass just 1 layer, argmin is 0. select_colors picks it. Correct.

    args = (
        final_hit_mask,
        final_depths,
        final_colors,
        world_to_camera,
        rays,
        *image_shape,
    )
    return postprocess(*args)


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
