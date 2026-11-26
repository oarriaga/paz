import jax
import jax.numpy as jp

import paz

from paz.graphics.renderer import rays, tiling

# TODO retune: measured best under 6k faces, but 2048 is 1.75x faster
# at 82k, so this is wrong for the room-scale meshes ahead.
FACE_CHUNK_SIZE = 128


def render(
    shape, intrinsics, pose, scene, mask, lights, tiles, chunk_size,
    shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    """Render `scene` into an `(H, W, 3)` image and an `(H, W)` depth map.

    `intrinsics` is a 3x3 OpenCV matrix with pixel centers on integer
    coordinates (see `paz.graphics.camera.build_rays`). The returned depth
    is z-depth along the camera -z axis, while the tracer itself works with
    distances along unit rays.

    # Arguments
    - `shape`: Tuple `(H, W)` with the image size in pixels.
    - `intrinsics`: Array of shape `(3, 3)`.
    - `pose`: Array of shape `(4, 4)`, the world to camera transform.
    - `scene`: `paz.graphics.Scene`.
    - `mask`: `None` to render every node, or one boolean per top-level
      scene node selecting the nodes to render.
    - `lights`: A `PointLight`, an `AreaLight` or a list of them.
    - `tiles`: Tuple `(H_tiles, W_tiles)`. The image is rendered in this
      many tiles; they must divide `H` and `W` exactly.
    - `chunk_size`: Int. Chunk size used when tracing the rays of a tile.
    - `shadows`: Bool passed to the ray tracer to enable shadows.
    - `shadow_mask`: `None` or a mask like `mask` used for shadows.
    - `num_bounces`: Int passed to the ray tracer.
    - `face_chunk_size`: Int. Chunk size used for mesh faces.

    # Returns
    Tuple `(image, depth)` with shapes `(H, W, 3)` and `(H, W)`.

    # Example
    ```python
    H, W = 32, 32
    intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 3, H, W)
    pose = paz.SE3.translation(jp.array([0.0, 0.0, -3.0]))
    scene = paz.graphics.Scene([paz.graphics.Sphere()])
    position = jp.array([2.0, 2.0, 2.0])
    light = paz.graphics.PointLight(jp.ones(3), position)
    args = (H, W), intrinsics, pose, scene, None, light, (1, 1), 1024
    image, depth = jax.jit(paz.lock(paz.graphics.render, *args))()
    image.shape, depth.shape  # (32, 32, 3), (32, 32)
    ```
    """
    compiled = paz.graphics.scene.compile(scene, lights, mask, shadow_mask)
    ray_args = compiled, shadows, num_bounces, face_chunk_size, chunk_size
    render_rays = paz.lock(rays.render_chunks, *ray_args)
    step_args = shape, intrinsics, pose, tiles, render_rays
    tile_step = paz.lock(tiling.render_step, *step_args)
    images, depths = tiling.scan(shape, tiles, tile_step)
    image = tiling.assemble(shape, tiles, images)
    depth = tiling.assemble(shape, tiles, depths)[..., 0]
    return image, depth


def render_masks(
    shape, intrinsics, pose, scene, lights, depth, tiles, chunk_size,
    num_objects=None, shadows=False, shadow_mask=None, num_bounces=1,
    face_chunk_size=FACE_CHUNK_SIZE,
):
    """Render one soft `(H, W, 1)` mask per scene node from its z-depth.

    `intrinsics` follows the same OpenCV convention as `render`.
    Each node is rendered alone with `render` and its depth is passed to
    `paz.depth.to_soft_mask`.

    # Arguments
    - `depth`: Tuple `(min_depth, max_depth)` given to
      `paz.depth.to_soft_mask`.
    - `num_objects`: Int or `None`. Masks are built for the first
      `num_objects` top-level nodes; `None` uses all of them.

    The remaining arguments are the ones of `render`.

    # Returns
    Array of shape `(num_objects, H, W, 1)` with values between 0 and 1.
    """
    if num_objects is None:
        num_objects = len(scene.nodes)
    min_depth, max_depth = depth
    object_masks = jp.eye(len(scene.nodes), dtype=bool)[:num_objects]
    args = shape, intrinsics, pose, scene, lights, tiles, chunk_size
    args += shadows, shadow_mask, num_bounces, face_chunk_size
    depths = jax.vmap(paz.lock(render_object_depth, *args))(object_masks)
    masks = paz.depth.to_soft_mask(depths, min_depth, max_depth)
    return jp.expand_dims(masks, axis=-1)


def render_object_depth(
    mask, shape, intrinsics, pose, scene, lights, tiles, chunk_size,
    shadows, shadow_mask, num_bounces, face_chunk_size,
):
    """Render the `(H, W)` z-depth map of the scene nodes selected by `mask`.

    It calls `render` with the same arguments and returns only its depth.
    """
    args = shape, intrinsics, pose, scene, mask, lights, tiles, chunk_size
    args += shadows, shadow_mask, num_bounces, face_chunk_size
    return render(*args)[1]
