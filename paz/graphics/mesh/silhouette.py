from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
from paz.graphics.mesh.intersect import EPSILON
from paz.graphics.mesh.tile import assemble
from paz.graphics.mesh.tile import assert_exact_tile_side
from paz.graphics.mesh.tile import make_tile_coordinates


Projection = namedtuple("Projection", "points depths")
Fragments = namedtuple("Fragments", "depths distances valid")

BLEND_EPSILON = 1e-4
FACES_PER_PIXEL = 50


def render_soft_mask(image_shape, pose, mesh, y_fov, sigma, chunk):
    H, W = image_shape
    pixels = build_pixel_coordinates(H, W)
    args = (pixels, H, W, pose, mesh, y_fov, sigma, chunk)
    mask = render_soft_mask_pixels(*args)
    return jp.reshape(mask, (H, W))


def tile_render_soft_mask(tile_shape, y_fov, H, W, pose, mesh, sigma, chunk):
    H_tiles, W_tiles = tile_shape
    assert_exact_tile_side(H, H_tiles)
    assert_exact_tile_side(W, W_tiles)
    args = (H, W, H_tiles, W_tiles, y_fov, pose, mesh, sigma, chunk)
    render_tile = partial(render_soft_mask_tile, *args)
    tile_coordinates = make_tile_coordinates(H_tiles, W_tiles)
    masks = jax.lax.scan(render_tile, None, tile_coordinates)[1]
    return assemble(H, W, H_tiles, W_tiles, masks)[..., 0]


def render_soft_mask_tile(*args):
    H, W, H_tiles, W_tiles, y_fov, pose, mesh, sigma, chunk = args[:9]
    carry, tile_arg = args[9:]
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    pixels = build_tile_pixel_coordinates(H, W, H_tiles, W_tiles, tile_arg)
    args = (pixels, H, W, pose, mesh, y_fov, sigma, chunk)
    mask = render_soft_mask_pixels(*args)
    mask = jp.reshape(mask, (tile_H, tile_W, 1))
    return carry, mask


def render_soft_mask_pixels(pixels, H, W, pose, mesh, y_fov, sigma, chunk):
    projection = project_mesh_vertices(mesh, pose, H, W, y_fov)
    faces, valid = pad_faces(mesh.faces, chunk)
    num_chunks = faces.shape[0] // chunk
    faces = jp.reshape(faces, (num_chunks, chunk, 3))
    valid = jp.reshape(valid, (num_chunks, chunk))
    init = build_empty_fragments(pixels.shape[0])
    blur = compute_blur_radius(sigma)
    step = partial(fragment_chunk_step, projection=projection, pixels=pixels)
    step = partial(step, blur=blur)
    fragments, _ = jax.lax.scan(step, init, (faces, valid))
    return blend_fragments(fragments.distances, fragments.valid, sigma)


def build_empty_fragments(num_pixels):
    shape = (num_pixels, FACES_PER_PIXEL)
    depths = jp.full(shape, jp.inf)
    distances = jp.zeros(shape)
    valid = jp.zeros(shape, dtype=bool)
    return Fragments(depths, distances, valid)


@jax.checkpoint
def fragment_chunk_step(fragments, data, projection, pixels, blur):
    faces, valid = data
    face_points = projection.points[faces]
    face_depths = projection.depths[faces]
    args = (face_points, face_depths, pixels, blur)
    distances, depths, candidates = compute_face_fragments(*args)
    candidates = jp.logical_and(candidates, valid[:, None])
    fragments = merge_fragments(fragments, distances, depths, candidates)
    return fragments, None


def compute_face_fragments(face_points, face_depths, pixels, blur):
    A, B, C = face_points[:, 0], face_points[:, 1], face_points[:, 2]
    barycentric, area = compute_barycentric_coordinates(A, B, C, pixels)
    inside = jp.all(barycentric > 0.0, axis=-1)
    clipped = clip_barycentric_coordinates(barycentric)
    depths = jp.sum(clipped * face_depths[:, None, :], axis=-1)
    distances = compute_triangle_distance(A, B, C, pixels)
    signed = jp.where(inside, -distances, distances)
    close = jp.logical_or(inside, distances < blur)
    candidates = jp.logical_and(valid_faces(area, face_depths)[:, None], close)
    candidates = jp.logical_and(candidates, depths > EPSILON)
    return signed, depths, candidates


def merge_fragments(fragments, distances, depths, valid):
    depths = jp.where(valid, depths, jp.inf).T
    distances = jp.where(valid, distances, 0.0).T
    valid = valid.T
    all_depths = jp.concatenate([fragments.depths, depths], axis=1)
    all_distances = jp.concatenate([fragments.distances, distances], axis=1)
    all_valid = jp.concatenate([fragments.valid, valid], axis=1)
    _, indices = jax.lax.top_k(-all_depths, FACES_PER_PIXEL)
    depths = jp.take_along_axis(all_depths, indices, axis=1)
    distances = jp.take_along_axis(all_distances, indices, axis=1)
    valid = jp.take_along_axis(all_valid, indices, axis=1)
    return Fragments(depths, distances, valid)


def blend_fragments(distances, valid, sigma):
    scale = jp.maximum(sigma, EPSILON)
    alpha = jax.nn.sigmoid(-distances / scale)
    alpha = jp.where(valid, alpha, 0.0)
    return 1.0 - jp.prod(1.0 - alpha, axis=1)


def compute_blur_radius(sigma):
    return jp.log(1.0 / BLEND_EPSILON - 1.0) * sigma


def compute_face_alpha(face_points, face_depths, pixels, sigma):
    blur = compute_blur_radius(sigma)
    args = (face_points, face_depths, pixels, blur)
    distances, _, valid = compute_face_fragments(*args)
    scale = jp.maximum(sigma, EPSILON)
    return jp.where(valid, jax.nn.sigmoid(-distances / scale), 0.0)


def project_mesh_vertices(mesh, pose, H, W, y_fov):
    world_vertices = paz.algebra.transform_points(mesh.transform, mesh.vertices)
    camera_vertices = paz.algebra.transform_points(pose, world_vertices)
    return project_camera_vertices(camera_vertices, H, W, y_fov)


def project_camera_vertices(vertices, H, W, y_fov):
    aspect = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(y_fov, aspect)
    depth = -vertices[:, 2]
    safe_depth = jp.where(jp.abs(depth) > EPSILON, depth, 1.0)
    plane_x = vertices[:, 0] / safe_depth
    plane_y = vertices[:, 1] / safe_depth
    scale = 2.0 / jp.minimum(H_world, W_world)
    points = jp.stack([scale * plane_x, scale * plane_y], axis=-1)
    return Projection(points, depth)


def compute_barycentric_coordinates(A, B, C, pixels):
    area = edge_function(C, A, B)
    safe_area = jp.where(jp.abs(area) > EPSILON, area, 1.0)
    pixel = pixels[None, :, :]
    w_A = edge_function(pixel, B[:, None, :], C[:, None, :])
    w_B = edge_function(pixel, C[:, None, :], A[:, None, :])
    w_C = edge_function(pixel, A[:, None, :], B[:, None, :])
    w_A = w_A / safe_area[:, None]
    w_B = w_B / safe_area[:, None]
    w_C = w_C / safe_area[:, None]
    return jp.stack([w_A, w_B, w_C], axis=-1), area


def clip_barycentric_coordinates(barycentric):
    barycentric = jp.maximum(barycentric, 0.0)
    scale = jp.maximum(jp.sum(barycentric, axis=-1, keepdims=True), EPSILON)
    return barycentric / scale


def compute_triangle_distance(A, B, C, pixels):
    distance_AB = compute_line_distance(A, B, pixels)
    distance_BC = compute_line_distance(B, C, pixels)
    distance_CA = compute_line_distance(C, A, pixels)
    return jp.minimum(distance_AB, jp.minimum(distance_BC, distance_CA))


def compute_line_distance(start, end, pixels):
    start = start[:, None, :]
    end = end[:, None, :]
    edge = end - start
    delta = pixels[None, :, :] - start
    length = jp.sum(edge * edge, axis=-1)
    scale = jp.sum(delta * edge, axis=-1) / jp.maximum(length, EPSILON)
    scale = jp.clip(scale, 0.0, 1.0)
    closest = start + scale[..., None] * edge
    distance = closest - pixels[None, :, :]
    return jp.sum(distance * distance, axis=-1)


def valid_faces(area, depths):
    valid_area = jp.abs(area) > EPSILON
    valid_depth = jp.all(depths > EPSILON, axis=1)
    return jp.logical_and(valid_area, valid_depth)


def edge_function(point, start, end):
    return cross2D(point - start, end - start)


def cross2D(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def pad_faces(faces, chunk):
    remainder = faces.shape[0] % chunk
    valid = jp.ones((faces.shape[0],), dtype=bool)
    if remainder == 0:
        return faces, valid
    pad = chunk - remainder
    faces = jp.concatenate([faces, jp.repeat(faces[-1:], pad, axis=0)])
    valid = jp.concatenate([valid, jp.zeros((pad,), dtype=bool)])
    return faces, valid


def build_pixel_coordinates(H, W, H_start=0, W_start=0):
    cols = jp.arange(W) + W_start + 0.5
    rows = jp.arange(H) + H_start + 0.5
    return build_coordinates(H, W, rows, cols)


def build_coordinates(H, W, rows, cols):
    col_grid, row_grid = jp.meshgrid(cols, rows)
    base = jp.minimum(H, W)
    x_grid = (col_grid - W / 2.0) * 2.0 / base
    y_grid = (H / 2.0 - row_grid) * 2.0 / base
    coords = [jp.ravel(x_grid), jp.ravel(y_grid)]
    return jp.stack(coords, axis=-1)


def build_tile_pixel_coordinates(H, W, H_tiles, W_tiles, tile_arg):
    W_tile_arg, H_tile_arg = tile_arg
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    H_start = tile_H * H_tile_arg
    W_start = tile_W * W_tile_arg
    cols = jp.arange(tile_W) + W_start + 0.5
    rows = jp.arange(tile_H) + H_start + 0.5
    return build_coordinates(H, W, rows, cols)
