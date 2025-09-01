# TODO render should take as argument name shapes not scene
# TODO check that depth information is computed properly when using mask
from functools import partial
import paz
import jax
import jax.numpy as jp

# from paz.graphics.scene import compile as compile_scene
from paz.graphics.constants import EPSILON
from paz.graphics.phong import (
    compute_colors,
    compute_colors_with_shadow,
)
from paz.graphics.shapes import intersection_cases, normal_cases


from paz.graphics.geometry import compute_points3D, transform_rays
from paz.backend.algebra import dot, normalize, transform_points


def to_color_image(hit_mask, colors, image_shape, background_color=1):
    H, W = image_shape
    image = jp.where(hit_mask, colors.T, background_color)
    image = image.reshape((3, H, W))
    image = jp.clip(image[:], 0, 255)
    image = jp.rollaxis(image, 0, 3)
    return image


def to_depth_image(
    hit_mask,
    depths,
    world_to_camera,
    ray_origins,
    ray_directions,
    image_shape,
    faraway=0,
):
    depths = jp.array(depths)
    H, W = image_shape
    depth_image = jp.min(depths, axis=0)
    depth_image = compute_points3D(ray_origins, ray_directions, depth_image)
    depth_image = transform_points(world_to_camera, depth_image)
    depth_image = -depth_image[:, -1]
    depth_image = jp.where(hit_mask, depth_image, faraway)
    depth_image = depth_image.reshape((1, *image_shape))
    depth_image = jp.rollaxis(depth_image, 0, 3)
    depth_image = depth_image[:, :, 0]
    return depth_image


def to_hit_image(hit_mask, image_shape):
    hit_image = hit_mask.reshape((1, *image_shape))
    hit_image = jp.rollaxis(hit_image, 0, 3)
    return hit_image


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


def invert_inside_normals(eyes, normals):
    inside_mask = dot(normals, eyes) < 0
    inside_mask = jp.expand_dims(inside_mask, 1)
    normals = jp.where(inside_mask, -normals, normals)
    return normals


def compute_eyes(ray_directions):
    eyes = -ray_directions
    return eyes


def move_toward_normals(points, normals, distance=EPSILON):
    return points + (normals * EPSILON)  # fixes pattern noise


def intersect_shape(shape, ray_origins, ray_directions):
    world_to_shape = jp.linalg.inv(shape.transform)
    rays_shape = transform_rays(world_to_shape, ray_origins, ray_directions)
    intersections = jax.lax.switch(shape.type, intersection_cases, *rays_shape)
    hit_mask, sorted_depths, depth = intersections
    # transform world points
    world_points = compute_points3D(ray_origins, ray_directions, depth)
    world_to_shape = jp.linalg.inv(shape.transform)
    shape_points = transform_points(world_to_shape, world_points)
    # transform world normals
    shape_normals = jax.lax.switch(shape.type, normal_cases, shape_points)
    world_normals = transform_points(world_to_shape.T, shape_normals)
    world_normals = normalize(world_normals)
    # postprocess normals
    eyes = compute_eyes(ray_directions)
    world_normals = invert_inside_normals(eyes, world_normals)
    world_points = move_toward_normals(world_points, world_normals)
    return hit_mask, depth, world_points, world_normals, eyes


def compute_scene_colors(shape, lights, points, normals, eyes):
    material = shape.material
    color = partial(compute_colors, shape, material, points, normals, eyes)
    scene_colors = jp.array([color(light) for light in lights])
    scene_colors = jp.sum(scene_colors, axis=0)
    return scene_colors


def render_shape(shape, lights, ray_origins, ray_directions):
    intersections = intersect_shape(shape, ray_origins, ray_directions)
    hit_mask, depth, points, normals, eyes = intersections
    colors = compute_scene_colors(shape, lights, points, normals, eyes)
    return hit_mask, depth, colors


def postprocess(
    hit_masks,
    depths,
    colors,
    world_to_camera,
    ray_origins,
    ray_directions,
    H,
    W,
):
    scene_hit_mask = compute_scene_hit_mask(hit_masks)
    scene_colors = select_colors(depths, colors)
    image = to_color_image(scene_hit_mask, scene_colors, (H, W))
    depth = to_depth_image(
        scene_hit_mask,
        depths,
        world_to_camera,
        ray_origins,
        ray_directions,
        (H, W),
    )
    return image, depth


def render(image_shape, world_to_camera, rays, scene, lights, mask=None):
    shapes, lights, mask = paz.graphics.scene.compile(scene, lights, mask)
    return _render(image_shape, world_to_camera, rays, shapes, lights, mask)


def render_shapes(shapes, lights, rays):
    hit_masks, depths, colors = [], [], []
    for shape in shapes:
        hit_mask, depth, color = render_shape(shape, lights, *rays)
        hit_masks.append(hit_mask)
        depths.append(depth)
        colors.append(color)
    return jp.array(hit_masks), jp.array(depths), jp.array(colors)


def _render(image_shape, world_to_camera, rays, shapes, lights, mask):
    # render_shapes = jax.vmap(paz.lock(render_shape, lights, *rays))
    # hit_masks, depths, colors = render_shapes(shapes)
    hit_masks, depths, colors = render_shapes(shapes, lights, rays)
    mask = jp.expand_dims(mask, 1)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(jp.expand_dims(mask, 1), depths, 1e6)
    return postprocess(
        hit_masks, depths, colors, world_to_camera, *rays, *image_shape
    )


def compute_shadow_mask(scene, masks, lights, rays, shape, mask):
    hit_mask, depth, points, normals, eyes = intersect_shape(shape, *rays)

    hit_mask = jp.where(mask, hit_mask, False)
    depth = jp.where(jp.expand_dims(mask, axis=1), depth, 1e6)
    points = jp.where(jp.expand_dims(mask, axis=1), points, 1e6)
    points = move_toward_normals(points, normals)

    shadow_masks = []
    for light in lights:
        points_to_light = light.position - points
        points_to_light_distances = jp.linalg.norm(
            points_to_light, axis=1, keepdims=True
        )
        points_to_light = points_to_light / points_to_light_distances
        intersect = partial(
            intersect_shape,
            ray_origins=points,
            ray_directions=points_to_light,
        )
        _hit_masks, scene_depths, scene_points3D, _normals, _ = jax.vmap(
            intersect, in_axes=0
        )(scene)

        # I have to mask all _hit_masks and depths using the generic mask
        _hit_masks = jp.where(jp.expand_dims(masks, axis=1), _hit_masks, False)
        scene_depths = jp.where(
            jp.expand_dims(masks, axis=[1, 2]), scene_depths, 1e6
        )

        scene_depth = jp.min(scene_depths, axis=0)
        scene_hit_mask = compute_scene_hit_mask(_hit_masks)
        points_to_light_distances = jp.squeeze(
            points_to_light_distances, axis=1
        )
        scene_depth = jp.squeeze(scene_depth, axis=1)
        first_hit_light_source = points_to_light_distances > scene_depth
        is_shadow = jp.logical_and(scene_hit_mask, first_hit_light_source)
        shadow_masks.append(is_shadow.astype(jp.int32))
    shadow_masks = jp.array(shadow_masks)
    return hit_mask, depth, shadow_masks, points, normals


def render_with_shadows(
    image_shape, world_to_camera, rays, shapes, lights, mask=None
):
    shapes, lights, mask = paz.graphics.scene.compile(shapes, lights, mask)
    render_args = (image_shape, world_to_camera, rays, shapes, lights, mask)
    return _render_with_shadows(*render_args)


def _render_with_shadows(
    image_shape, world_to_camera, rays, shapes, lights, mask
):
    mask_args = (shapes, mask, lights, rays)
    compute_shadow_masks = jax.vmap(partial(compute_shadow_mask, *mask_args))
    mask = jp.expand_dims(mask, axis=1)
    hit_masks, depths, shadow_masks, points, normals = compute_shadow_masks(
        shapes, mask
    )
    eyes = compute_eyes(rays[1])

    def color(shape, points, normals, shadow_mask):
        color_by_lights = []
        for light_arg, light in enumerate(lights):
            shadow_light_mask = shadow_mask[light_arg, :]
            color = compute_colors_with_shadow(
                shape,
                shape.material,
                points,
                normals,
                eyes,
                light,
                shadow_light_mask,
            )
            color_by_lights.append(color)
        return jp.array(color_by_lights)

    colors_per_light = jax.vmap(color)(shapes, points, normals, shadow_masks)
    colors = jp.sum(colors_per_light, axis=1)
    return postprocess(
        hit_masks, depths, colors, world_to_camera, *rays, *image_shape
    )
