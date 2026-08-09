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
    masks = []
    for object_arg in range(num_objects):
        mask = build_object_mask(len(scene.nodes), object_arg)
        args = shape, y_FOV, pose, scene, mask, lights, tiles, chunk_size
        args += shadows, shadow_mask, num_bounces, face_chunk_size
        _, depth_image = render(*args)
        soft = paz.depth.to_soft_mask(depth_image, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def build_object_mask(num_nodes, object_arg):
    return jp.zeros((num_nodes,), dtype=bool).at[object_arg].set(True)
