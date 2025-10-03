import jax.numpy as jp
import paz
import jax
from paz.graphics import EPSILON, FARAWAY


def render_shapes(shapes, lights, rays):
    hit_masks, depths, colors = [], [], []
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    for image_size, shape_group in grouped_shapes.items():
        shape_group = paz.graphics.shapes.merge(*shape_group)
        render_shapes = jax.vmap(paz.lock(render_shape, lights, rays))
        group_hit_masks, group_depths, group_colors = render_shapes(shape_group)
        hit_masks.append(group_hit_masks)
        depths.append(group_depths)
        colors.append(group_colors)
    hit_masks = jp.concatenate(hit_masks, axis=0)
    depths = jp.concatenate(depths, axis=0)
    colors = jp.concatenate(colors, axis=0)
    return hit_masks, depths, colors


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


def _render(image_shape, world_to_camera, rays, shapes, lights, mask):
    hit_masks, depths, colors = render_shapes(shapes, lights, rays)
    mask = jp.expand_dims(mask, 1)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(jp.expand_dims(mask, 1), depths, 1e6)  # use FARAWAY?
    args = (hit_masks, depths, colors, world_to_camera, rays, *image_shape)
    return postprocess(*args)


def intersect_groups(shapes, origins, directions):
    grouped_shapes = paz.graphics.shapes.group_by_pattern_size(shapes)
    hit_masks, depths, points, normals, eyes = [], [], [], [], []
    intersect = paz.lock(paz.graphics.shapes.intersect, origins, directions)
    for image_size, group in grouped_shapes.items():
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
    return hit_masks, depths, points, normals, eyes


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


def _render_with_shadows(img_size, world_to_camera, rays, shapes, lights, mask):
    hit_masks, depths, points, normals, eyes = intersect_groups(shapes, *rays)
    mask = jp.expand_dims(mask, 1)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(jp.expand_dims(mask, axis=1), depths, FARAWAY)
    points = jp.where(jp.expand_dims(mask, axis=1), points, FARAWAY)
    points = paz.pointcloud.move_along_normals(points, normals, EPSILON)
    closest_points = get_closest_hit_points(depths, points)
    color_shapes = []
    for light in lights:
        points_to_light = light.position - closest_points
        points_to_light_norms = paz.algebra.compute_norms(points_to_light, 1)
        points_to_light = points_to_light / points_to_light_norms
        shadows = intersect_groups(shapes, closest_points, points_to_light)[:2]
        points_to_light_hit_masks, points_to_light_depths = shadows

        # points_to_light_hit_masks = jp.where(
        #     jp.expand_dims(mask, axis=1), points_to_light_hit_masks, False
        # )
        points_to_light_hit_masks = jp.where(
            mask, points_to_light_hit_masks, False
        )
        points_to_light_hit_mask = compute_scene_hit_mask(
            points_to_light_hit_masks
        )
        points_to_light_depths = jp.where(
            jp.expand_dims(mask, axis=2), points_to_light_depths, FARAWAY
        )
        points_to_light_depth = jp.min(points_to_light_depths, axis=0)
        points_to_light_depth = jp.squeeze(points_to_light_depth, axis=1)
        points_to_light_norms = jp.squeeze(points_to_light_norms, axis=1)
        first_hit_light_source = points_to_light_norms > points_to_light_depth

        is_shadow = jp.logical_and(
            points_to_light_hit_mask, first_hit_light_source
        )

        _, _, colors = _render_shapes(shapes, [light], rays, is_shadow)
        color_shapes.append(colors)

        # shadow_masks.append(is_shadow.astype(jp.int32))
    # shadow_masks = jp.array(shadow_masks)
    # print("shadow_masks", shadow_masks.shape)
    # shadow_mask = jp.sum(shadow_masks[0], axis=0).astype(bool)
    # shadow_mask = jp.expand_dims(shadow_mask, axis=0)
    # print("shadow_masks", shadow_masks.shape)
    # # shadow_mask = jp.sum(shadow_masks, axis=[0, 1]).astype(bool)
    # hit_masks, depths, colors = _render_shapes(
    # shapes, lights, rays, shadow_mask
    # )
    # mask = jp.expand_dims(mask, 1)
    colors = jp.array(color_shapes)
    colors = jp.sum(colors, axis=0)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(jp.expand_dims(mask, 1), depths, FARAWAY)  # use FARAWAY?
    args = (hit_masks, depths, colors, world_to_camera, rays, *img_size)
    return postprocess(*args)


def render(image_shape, world_to_camera, rays, scene, lights, mask, shadows):
    shapes, lights, mask = paz.graphics.scene.compile(scene, lights, mask)
    args = (image_shape, world_to_camera, rays, shapes, lights, mask)
    return _render_with_shadows(*args) if shadows else _render(*args)
