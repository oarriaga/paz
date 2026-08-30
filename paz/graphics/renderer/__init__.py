import jax
import jax.numpy as jp

import paz

from paz.graphics.renderer import rays, tiling

# TODO retune: measured best under 6k faces, but 2048 is 1.75x faster
# at 82k, so this is wrong for the room-scale meshes ahead.
FACE_CHUNK_SIZE = 128


def render(
    shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size,
    shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    compiled = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)
    ray_args = compiled, shadows, num_bounces, face_chunk_size, chunk_size
    render_rays = paz.lock(rays.render_chunks, *ray_args)
    step_args = shape, y_FOV, pose, tiles, render_rays
    tile_step = paz.lock(tiling.render_step, *step_args)
    images, depths = tiling.scan(shape, tiles, tile_step)
    image = tiling.assemble(shape, tiles, images)
    depth = tiling.assemble(shape, tiles, depths)[..., 0]
    return image, depth


def render_masks(
    shape, y_FOV, pose, scene, lights, depth, tiles, chunk_size,
    num_objects=None, shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    if num_objects is None:
        num_objects = len(scene.nodes)
    min_depth, max_depth = depth
    object_masks = jp.eye(len(scene.nodes), dtype=bool)[:num_objects]
    args = shape, y_FOV, pose, scene, lights, tiles, chunk_size
    args += shadows, shadow_mask, num_bounces, face_chunk_size
    depths = jax.vmap(paz.lock(render_object_depth, *args))(object_masks)
    masks = paz.depth.to_soft_mask(depths, min_depth, max_depth)
    return jp.expand_dims(masks, axis=-1)


def render_object_depth(
    mask, shape, y_FOV, pose, scene, lights, tiles, chunk_size,
    shadows, shadow_mask, num_bounces, face_chunk_size,
):
    args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
    args += shadows, shadow_mask, num_bounces, face_chunk_size
    return render(*args)[1]
