import jax.numpy as jp
import paz


def build_rays(size, intrinsics, world_to_camera=None):
    """Build one world-space ray per pixel of an `(H, W)` image.

    `intrinsics` is a 3x3 OpenCV matrix: the center of pixel (column i,
    row j) is at (i, j), as in `paz.pointcloud.from_depth`. Returns
    `(origins, directions)` with unit directions, in row-major pixel order.
    """
    if world_to_camera is None:
        world_to_camera = jp.eye(4)
    H, W = size[:2]
    pixels = build_pixel_grid(H, W)
    return build_pixel_rays(intrinsics, pixels, world_to_camera)


def build_pixel_grid(H, W):
    u_grid, v_grid = jp.meshgrid(jp.arange(W), jp.arange(H))
    return jp.stack([u_grid.ravel(), v_grid.ravel()], axis=-1)


def build_pixel_rays(intrinsics, pixels, world_to_camera):
    """Build unit world-space rays through `pixels`, an `(N, 2)` `(u, v)`.

    The camera looks along -z with y up, so the OpenCV image axes are
    flipped here; `paz.graphics.mesh.silhouette.project_to_pixels` undoes
    the same flip when it projects points back to pixels.
    """
    x = (pixels[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1]
    directions = jp.stack([x, -y, -jp.ones_like(x)], axis=-1)
    origins = jp.zeros_like(directions)
    camera_to_world = jp.linalg.inv(world_to_camera)
    args = camera_to_world, origins, directions
    origins, directions = paz.algebra.transform_rays(*args)
    return origins, paz.algebra.normalize(directions)


def compute_intrinsics(y_FOV, H, W):
    """Build the 3x3 OpenCV intrinsics of a centered square-pixel camera.

    The principal point is `((W - 1) / 2, (H - 1) / 2)` because pixel
    centers sit on integer coordinates.
    """
    focal_length = compute_focal_length(y_FOV) * H / 2.0
    x_center = (W - 1) / 2.0
    y_center = (H - 1) / 2.0
    rows = [[focal_length, 0.0, x_center], [0.0, focal_length, y_center]]
    return jp.array(rows + [[0.0, 0.0, 1.0]])


def compute_focal_length(y_field_of_view):
    return 1 / jp.tan(y_field_of_view / 2.0)
