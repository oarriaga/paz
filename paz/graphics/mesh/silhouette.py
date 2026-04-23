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
BinArgs = namedtuple("BinArgs", "shape max_faces")

BLEND_EPSILON = 1e-4
FACES_PER_PIXEL = 50


def unpack_bin_shape(bins):
    """Convert the public bin shape into `(H_bin, W_bin)`.

    # Arguments
    - `bins`: `BinArgs` with `shape` and `max_faces`.
    - `bins.shape`: Either `int` or `(H_bin, W_bin)`.

    # Returns
    - Tuple `(H_bin, W_bin)` of Python or JAX scalar bin sizes.
    """
    shape = bins.shape
    try:
        H_bin, W_bin = shape
    except TypeError:
        return shape, shape
    except ValueError as error:
        raise ValueError("bin shape must be int or (H_bin, W_bin).") from error
    return H_bin, W_bin


def tile_render_binned_soft_mask(bins, y_fov, H, W, pose, mesh, sigma, chunk):
    """Render a differentiable silhouette mask with spatial face bins.

    The image is split into bins. Each face is assigned only to bins touched
    by its projected bounding box expanded by the blur radius. Each bin then
    renders only its assigned faces, preserving the SoftRas-style mask while
    avoiding dense face-pixel work.

    # Arguments
    - `bins`: `BinArgs` with `shape` and `max_faces` per bin.
    - `bins.shape`: Either `int` for square bins or `(H_bin, W_bin)`.
    - `y_fov`: Python or JAX scalar camera vertical field of view in radians.
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `pose`: JAX array with shape `(4, 4)`, world-to-camera transform.
    - `mesh`: `Mesh` with `vertices` `(V, 3)` and `faces` `(F, 3)`,
      where `V` is the number of vertices and `F` is the number of faces.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.
    - `chunk`: Python int number of faces processed per fragment step.

    # Returns
    - JAX array with shape `(H, W)`, soft silhouette mask in `[0, 1]`.
    """
    H_bin, W_bin = unpack_bin_shape(bins)
    assert_exact_tile_side(H, H_bin)
    assert_exact_tile_side(W, W_bin)
    H_bins, W_bins = H // H_bin, W // W_bin
    projection = project_mesh_vertices(mesh, pose, H, W, y_fov)
    args = (projection, mesh.faces, H, W, sigma, bins)
    bin_faces, bin_valid = build_bin_faces(*args)
    args = (H, W, H_bins, W_bins, projection, sigma, chunk)
    args = args + (bin_faces, bin_valid)
    render_bin = partial(render_binned_soft_mask_tile, *args)
    bin_coordinates = make_tile_coordinates(H_bins, W_bins)
    masks = jax.lax.scan(render_bin, None, bin_coordinates)[1]
    return assemble(H, W, H_bins, W_bins, masks)[..., 0]


def render_binned_soft_mask_tile(*args):
    """Render one bin tile of the binned soft silhouette mask.

    This is the `jax.lax.scan` body used by `tile_render_binned_soft_mask`.
    It builds pixel coordinates for one bin, selects that bin's compact face
    list, and skips all math when the bin has no valid face slots.

    # Arguments
    - `args[0:7]`: `H`, `W`, `H_bins`, `W_bins`, `projection`, `sigma`,
      and `chunk`.
    - `args[7]`: JAX int array with shape `(B, M, 3)`, faces per bin.
    - `args[8]`: JAX bool array with shape `(B, M)`, valid face slots.
    - `args[9]`: scan carry, unused and usually `None`.
    - `args[10]`: JAX int array with shape `(2,)`, `(x_bin, y_bin)`.

    Here `B = H_bins * W_bins` is the number of bins, and `M` is the fixed
    face capacity per bin, `min(bins.max_faces, F)`.

    # Returns
    - Tuple `(None, mask)` where `mask` has shape `(Hb, Wb, 1)`.
      `Hb = H // H_bins` and `Wb = W // W_bins`.
    """
    H, W, H_bins, W_bins, projection, sigma, chunk = args[:7]
    bin_faces, bin_valid, carry, bin_arg = args[7:]
    bin_id = bin_arg[1] * W_bins + bin_arg[0]
    pixels = build_tile_pixel_coordinates(H, W, H_bins, W_bins, bin_arg)
    tile_shape = (bin_side(H, H_bins), bin_side(W, W_bins), 1)
    data = (bin_faces[bin_id], bin_valid[bin_id], pixels)
    run_bin = partial(render_active_bin, projection=projection, sigma=sigma)
    run_bin = partial(run_bin, chunk=chunk, tile_shape=tile_shape)
    empty_bin = partial(render_empty_bin, tile_shape=tile_shape)
    active = jp.any(bin_valid[bin_id])
    return jax.lax.cond(active, run_bin, empty_bin, data)


def render_active_bin(data, projection, sigma, chunk, tile_shape):
    """Render a non-empty bin from its local face list.

    The face list is padded to a multiple of `chunk`, scanned in chunks, and
    merged into the nearest `FACES_PER_PIXEL` fragments per pixel before the
    fragments are alpha-blended into a soft mask.

    # Arguments
    - `data`: Tuple `(bin_faces, bin_valid, pixels)`.
    - `data[0]`: JAX int array with shape `(M, 3)`, face vertex indices.
    - `data[1]`: JAX bool array with shape `(M,)`, valid face slots.
    - `data[2]`: JAX array with shape `(P, 2)`, normalized pixel centers.
    - `projection`: `Projection` with points `(V, 2)` and depths `(V,)`.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.
    - `chunk`: Python int number of faces per scanned chunk.
    - `tile_shape`: Python tuple `(Hb, Wb, 1)` for the output tile.

    `M` is the fixed face capacity per bin, `P = Hb * Wb`, and `V` is the
    number of projected mesh vertices.

    # Returns
    - Tuple `(None, mask)` where `mask` is a JAX array of `tile_shape`.
    """
    bin_faces, bin_valid, pixels = data
    faces, valid = pad_binned_faces(bin_faces, bin_valid, chunk)
    num_chunks = faces.shape[0] // chunk
    faces = jp.reshape(faces, (num_chunks, chunk, 3))
    valid = jp.reshape(valid, (num_chunks, chunk))
    fragments = build_empty_fragments(pixels.shape[0])
    blur = compute_blur_radius(sigma)
    step = partial(fragment_chunk_or_empty_step, projection=projection)
    step = partial(step, pixels=pixels)
    step = partial(step, blur=blur)
    fragments, _ = jax.lax.scan(step, fragments, (faces, valid))
    mask = blend_fragments(fragments.distances, fragments.valid, sigma)
    mask = jp.reshape(mask, tile_shape)
    return None, mask


def render_empty_bin(data, tile_shape):
    """Return an all-zero mask for a bin with no valid face slots.

    This branch avoids running fragment math for bins that cannot affect the
    silhouette because no projected face overlaps their expanded bounds.

    # Arguments
    - `data`: Unused tuple passed by `jax.lax.cond`.
    - `tile_shape`: Python tuple `(Hb, Wb, 1)` for the output tile.

    # Returns
    - Tuple `(None, mask)` where `mask` is zeros with shape `tile_shape`.
    """
    return None, jp.zeros(tile_shape)


def fragment_chunk_or_empty_step(fragments, data, projection, pixels, blur):
    """Run a face chunk only when it contains at least one valid slot.

    Empty chunks are produced by fixed-size bin padding. Skipping them avoids
    expensive face-pixel distance computation while preserving JAX shapes.

    # Arguments
    - `fragments`: `Fragments` with arrays of shape `(P, K)`.
    - `data`: Tuple `(faces, valid)` for one chunk.
    - `data[0]`: JAX int array with shape `(C, 3)`.
    - `data[1]`: JAX bool array with shape `(C,)`.
    - `projection`: `Projection` with points `(V, 2)` and depths `(V,)`.
    - `pixels`: JAX array with shape `(P, 2)`.
    - `blur`: Python or JAX scalar squared-distance candidate radius.

    `P` is the number of pixels in this bin, `K = FACES_PER_PIXEL`,
    `C = chunk`, and `V` is the number of projected mesh vertices.

    # Returns
    - Tuple `(fragments, None)` with unchanged or updated fragments.
    """
    def run_fragment_chunk(args):
        return fragment_chunk_step(*args)

    def skip_fragment_chunk(args):
        return args[0], None

    args = (fragments, data, projection, pixels, blur)
    active = jp.any(data[1])
    return jax.lax.cond(active, run_fragment_chunk, skip_fragment_chunk, args)


@jax.checkpoint
def fragment_chunk_step(fragments, data, projection, pixels, blur):
    """Merge one face chunk into the current per-pixel fragments.

    This checkpointed scan body computes candidate signed distances and
    depths for one chunk, masks invalid padded faces, and keeps the closest
    `FACES_PER_PIXEL` fragments per pixel.

    # Arguments
    - `fragments`: `Fragments` with arrays of shape `(P, K)`.
    - `data`: Tuple `(faces, valid)` for one chunk.
    - `data[0]`: JAX int array with shape `(C, 3)`.
    - `data[1]`: JAX bool array with shape `(C,)`.
    - `projection`: `Projection` with points `(V, 2)` and depths `(V,)`.
    - `pixels`: JAX array with shape `(P, 2)`.
    - `blur`: Python or JAX scalar squared-distance candidate radius.

    `P` is the number of pixels in this bin, `K = FACES_PER_PIXEL`,
    `C = chunk`, and `V` is the number of projected mesh vertices.

    # Returns
    - Tuple `(fragments, None)` with updated `Fragments` arrays `(P, K)`.
    """
    faces, valid = data
    face_points = projection.points[faces]
    face_depths = projection.depths[faces]
    args = (face_points, face_depths, pixels, blur)
    distances, depths, candidates = compute_face_fragments(*args)
    candidates = jp.logical_and(candidates, valid[:, None])
    fragments = merge_fragments(fragments, distances, depths, candidates)
    return fragments, None


def compute_face_fragments(face_points, face_depths, pixels, blur):
    """Compute signed distances, depths, and validity for face-pixel pairs.

    For every face and pixel, the function computes barycentric coordinates,
    interpolated depth, squared distance to the bounded triangle outline, and
    a validity mask. Inside pixels get negative distances; outside pixels get
    positive distances if they lie within the blur radius.

    # Arguments
    - `face_points`: JAX array with shape `(F, 3, 2)`.
    - `face_depths`: JAX array with shape `(F, 3)`.
    - `pixels`: JAX array with shape `(P, 2)`.
    - `blur`: Python or JAX scalar squared-distance candidate radius.

    `F` is the number of faces in the current chunk, and `P` is the number
    of pixels in the current bin.

    # Returns
    - `signed`: JAX array with shape `(F, P)`.
    - `depths`: JAX array with shape `(F, P)`.
    - `candidates`: JAX bool array with shape `(F, P)`.
    """
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
    """Keep the nearest valid fragments after adding one candidate chunk.

    Candidate arrays arrive as face-major `(F, P)` values. They are converted
    to pixel-major `(P, F)`, concatenated with the current `(P, K)` fragment
    buffers, and reduced back to `(P, K)` by nearest depth.

    # Arguments
    - `fragments`: Current `Fragments` with arrays of shape `(P, K)`.
    - `distances`: JAX array with shape `(F, P)`.
    - `depths`: JAX array with shape `(F, P)`.
    - `valid`: JAX bool array with shape `(F, P)`.

    `F` is the number of faces in the current chunk, `P` is the number of
    pixels in this bin, and `K = FACES_PER_PIXEL`.

    # Returns
    - `Fragments` with `depths`, `distances`, and `valid` shape `(P, K)`.
    """
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
    """Blend per-pixel face fragments into a soft silhouette mask.

    Each valid signed squared distance becomes an alpha value with
    `sigmoid(-distance / sigma)`. The final mask is the probability that at
    least one candidate face covers the pixel.

    # Arguments
    - `distances`: JAX array with shape `(P, K)`, signed distances.
    - `valid`: JAX bool array with shape `(P, K)`, valid fragment slots.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.

    `P` is the number of pixels in the bin and `K = FACES_PER_PIXEL`.

    # Returns
    - JAX array with shape `(P,)`, blended mask values in `[0, 1]`.
    """
    scale = jp.maximum(sigma, EPSILON)
    alpha = jax.nn.sigmoid(-distances / scale)
    alpha = jp.where(valid, alpha, 0.0)
    return 1.0 - jp.prod(1.0 - alpha, axis=1)


def build_empty_fragments(num_pixels):
    """Create an empty per-pixel fragment buffer.

    The buffer stores the nearest `FACES_PER_PIXEL` candidate fragments for
    each pixel. Empty slots have infinite depth and invalid masks.

    # Arguments
    - `num_pixels`: Python int `P`, the number of pixels in the bin.

    # Returns
    - `Fragments` with `depths` `(P, K)`, `distances` `(P, K)`,
      and `valid` `(P, K)`, where `K = FACES_PER_PIXEL`.
    """
    shape = (num_pixels, FACES_PER_PIXEL)
    depths = jp.full(shape, jp.inf)
    distances = jp.zeros(shape)
    valid = jp.zeros(shape, dtype=bool)
    return Fragments(depths, distances, valid)


def count_binned_faces(image_shape, pose, mesh, y_fov, sigma, bins):
    """Count how many projected faces overlap each image bin.

    This helper is used before rendering to choose or validate
    `bins.max_faces`. It performs the same bin-overlap test used by the
    renderer, but returns counts instead of padded face lists.

    # Arguments
    - `image_shape`: Python tuple `(H, W)` with image size in pixels.
    - `pose`: JAX array with shape `(4, 4)`, world-to-camera transform.
    - `mesh`: `Mesh` with `vertices` `(V, 3)` and `faces` `(F, 3)`.
    - `y_fov`: Python or JAX scalar camera vertical field of view in radians.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.
    - `bins`: `BinArgs` with `shape` and `max_faces` per bin.

    `V` is the number of mesh vertices, `F` is the number of mesh faces, and
    `B = (H // H_bin) * (W // W_bin)` after unpacking `bins.shape`.

    # Returns
    - JAX int array with shape `(B,)`, overlapping face count per bin.
    """
    H, W = image_shape
    projection = project_mesh_vertices(mesh, pose, H, W, y_fov)
    face_points = projection.points[mesh.faces]
    face_depths = projection.depths[mesh.faces]
    overlap = compute_bin_overlaps(face_points, face_depths, H, W, sigma, bins)
    return jp.sum(overlap, axis=1)


def build_bin_faces(projection, faces, H, W, sigma, bins):
    """Build fixed-size face lists for every image bin.

    The function computes a boolean face-bin overlap matrix and keeps the
    first `bins.max_faces` overlapping faces per bin with `jax.lax.top_k`.
    Fixed-size arrays keep the renderer compatible with JAX compilation.

    # Arguments
    - `projection`: `Projection` with points `(V, 2)` and depths `(V,)`.
    - `faces`: JAX int array with shape `(F, 3)`, mesh face indices.
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.
    - `bins`: `BinArgs` with `shape` and `max_faces` per bin.

    `V` is the number of mesh vertices, `F` is the number of faces,
    `B = (H // H_bin) * (W // W_bin)` after unpacking `bins.shape`, and
    `M = min(bins.max_faces, F)`.

    # Returns
    - `bin_faces`: JAX int array with shape `(B, M, 3)`.
    - `bin_valid`: JAX bool array with shape `(B, M)`.
    """
    max_faces = min(bins.max_faces, faces.shape[0])
    face_points = projection.points[faces]
    face_depths = projection.depths[faces]
    overlap = compute_bin_overlaps(face_points, face_depths, H, W, sigma, bins)
    face_args = jp.arange(faces.shape[0])
    scores = jp.where(overlap, faces.shape[0] - face_args[None, :], -1)
    _, indices = jax.lax.top_k(scores, max_faces)
    valid = jp.take_along_axis(overlap, indices, axis=1)
    return faces[indices], valid


def pad_binned_faces(faces, valid, chunk):
    """Pad a bin face list so it can be reshaped into face chunks.

    Padded entries repeat the last face but are marked invalid. In this
    context, invalid means the slot should not contribute to fragments.

    # Arguments
    - `faces`: JAX int array with shape `(M, 3)`, face vertex indices.
    - `valid`: JAX bool array with shape `(M,)`, valid face slots.
    - `chunk`: Python int target chunk size.

    `M` is the fixed face capacity per bin.

    # Returns
    - `faces`: JAX int array with shape `(C * chunk, 3)`.
    - `valid`: JAX bool array with shape `(C * chunk,)`.
      `C = ceil(M / chunk)` when padding is needed.
    """
    remainder = faces.shape[0] % chunk
    if remainder == 0:
        return faces, valid
    pad = chunk - remainder
    faces = jp.concatenate([faces, jp.repeat(faces[-1:], pad, axis=0)])
    valid = jp.concatenate([valid, jp.zeros((pad,), dtype=bool)])
    return faces, valid


def compute_bin_overlaps(face_points, face_depths, H, W, sigma, bins):
    """Compute which projected faces can influence each image bin.

    A face overlaps a bin when its projected bounding box, expanded by the
    blur radius, intersects the bin bounds and the face is valid. A projected
    face is valid when it has non-zero area and all vertices have positive
    camera depth.

    # Arguments
    - `face_points`: JAX array with shape `(F, 3, 2)`.
    - `face_depths`: JAX array with shape `(F, 3)`.
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.
    - `bins`: `BinArgs` with `shape` and `max_faces` per bin.

    `F` is the number of projected faces and
    `B = (H // H_bin) * (W // W_bin)` after unpacking `bins.shape`.

    # Returns
    - JAX bool array with shape `(B, F)`, true for face-bin overlap.
    """
    x_min, x_max, y_min, y_max = compute_face_boxes(face_points, sigma)
    bin_boxes = build_bin_boxes(H, W, unpack_bin_shape(bins))
    bin_x_min, bin_x_max, bin_y_min, bin_y_max = bin_boxes
    left = x_min[None] <= bin_x_max[:, None]
    right = bin_x_min[:, None] <= x_max[None]
    x_hit = jp.logical_and(left, right)
    lower = y_min[None] <= bin_y_max[:, None]
    upper = bin_y_min[:, None] <= y_max[None]
    y_hit = jp.logical_and(lower, upper)
    valid = compute_valid_projected_faces(face_points, face_depths)
    return jp.logical_and(jp.logical_and(x_hit, y_hit), valid[None])


def compute_face_boxes(face_points, sigma):
    """Compute expanded 2D bounding boxes for projected triangles.

    Each projected triangle gets an axis-aligned box in normalized image
    coordinates. The box is expanded by the square-root blur radius because
    stored distances are squared.

    # Arguments
    - `face_points`: JAX array with shape `(F, 3, 2)`.
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.

    `F` is the number of projected faces.

    # Returns
    - `x_min`: JAX array with shape `(F,)`.
    - `x_max`: JAX array with shape `(F,)`.
    - `y_min`: JAX array with shape `(F,)`.
    - `y_max`: JAX array with shape `(F,)`.
    """
    radius = jp.sqrt(compute_blur_radius(sigma))
    x, y = face_points[..., 0], face_points[..., 1]
    x_min, x_max = jp.min(x, axis=1) - radius, jp.max(x, axis=1) + radius
    y_min, y_max = jp.min(y, axis=1) - radius, jp.max(y, axis=1) + radius
    return x_min, x_max, y_min, y_max


def compute_valid_projected_faces(face_points, face_depths):
    """Mark projected faces with non-zero area and positive depth.

    An invalid projected face is degenerate in screen space or has at least
    one vertex behind or too close to the camera. Invalid faces are excluded
    from bin overlap lists and do not contribute to the soft mask.

    # Arguments
    - `face_points`: JAX array with shape `(F, 3, 2)`.
    - `face_depths`: JAX array with shape `(F, 3)`.

    `F` is the number of projected faces.

    # Returns
    - JAX bool array with shape `(F,)`, true for renderable faces.
    """
    A, B, C = face_points[:, 0], face_points[:, 1], face_points[:, 2]
    area = edge_function(C, A, B)
    return valid_faces(area, face_depths)


def valid_faces(area, depths):
    """Mark faces that have non-zero projected area and positive depth.

    A face is valid only when its projected area is not degenerate and all
    three vertices are in front of the camera.

    # Arguments
    - `area`: JAX array with shape `(F,)`, signed twice-area.
    - `depths`: JAX array with shape `(F, 3)`, face vertex depths.

    `F` is the number of projected faces.

    # Returns
    - JAX bool array with shape `(F,)`.
    """
    valid_area = jp.abs(area) > EPSILON
    valid_depth = jp.all(depths > EPSILON, axis=1)
    return jp.logical_and(valid_area, valid_depth)


def project_mesh_vertices(mesh, pose, H, W, y_fov):
    """Project mesh vertices to normalized image-plane coordinates.

    The mesh transform is applied first, then the camera pose. Camera-space
    vertices are perspective-projected into the normalized 2D coordinates
    used by pixel centers and bin boxes.

    # Arguments
    - `mesh`: `Mesh` with `vertices` `(V, 3)` and `transform` `(4, 4)`.
    - `pose`: JAX array with shape `(4, 4)`, world-to-camera transform.
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `y_fov`: Python or JAX scalar camera vertical field of view in radians.

    `V` is the number of mesh vertices.

    # Returns
    - `Projection` with `points` `(V, 2)` and `depths` `(V,)`.
    """
    world_vertices = paz.algebra.transform_points(mesh.transform, mesh.vertices)
    camera_vertices = paz.algebra.transform_points(pose, world_vertices)
    return project_camera_vertices(camera_vertices, H, W, y_fov)


def project_camera_vertices(vertices, H, W, y_fov):
    """Project camera-space vertices to the normalized image plane.

    Depth is positive in front of the camera because the camera looks along
    negative z. The x and y coordinates are divided by depth and scaled by
    the field of view and aspect ratio.

    # Arguments
    - `vertices`: JAX array with shape `(V, 3)`, camera-space vertices.
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `y_fov`: Python or JAX scalar camera vertical field of view in radians.

    `V` is the number of vertices.

    # Returns
    - `Projection` with `points` `(V, 2)` and `depths` `(V,)`.
    """
    aspect = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(y_fov, aspect)
    depth = -vertices[:, 2]
    safe_depth = jp.where(jp.abs(depth) > EPSILON, depth, 1.0)
    plane_x = vertices[:, 0] / safe_depth
    plane_y = vertices[:, 1] / safe_depth
    scale = 2.0 / jp.minimum(H_world, W_world)
    points = jp.stack([scale * plane_x, scale * plane_y], axis=-1)
    return Projection(points, depth)


def compute_blur_radius(sigma):
    """Convert sigmoid softness into a squared-distance candidate radius.

    Distances farther than this radius have alpha below `BLEND_EPSILON`, so
    they can be safely ignored during fragment candidate selection.

    # Arguments
    - `sigma`: Python or JAX scalar controlling sigmoid mask softness.

    # Returns
    - Python or JAX scalar squared-distance threshold.
    """
    return jp.log(1.0 / BLEND_EPSILON - 1.0) * sigma


def build_bin_boxes(H, W, bin_shape):
    """Build normalized image-plane bounds for all bins.

    Pixel-space bin edges are converted into the same normalized coordinates
    used by projected mesh vertices. The returned arrays are flattened in
    row-major bin order.

    # Arguments
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `bin_shape`: Tuple `(H_bin, W_bin)` in pixels.

    `B = (H // H_bin) * (W // W_bin)` is the number of bins.

    # Returns
    - `x_min`: JAX array with shape `(B,)`.
    - `x_max`: JAX array with shape `(B,)`.
    - `y_min`: JAX array with shape `(B,)`.
    - `y_max`: JAX array with shape `(B,)`.
    """
    H_bin, W_bin = bin_shape
    x_min, x_max = build_x_bin_bounds(H, W, W_bin)
    y_min, y_max = build_y_bin_bounds(H, W, H_bin)
    x_min, y_min = jp.meshgrid(x_min, y_min)
    x_max, y_max = jp.meshgrid(x_max, y_max)
    return jp.ravel(x_min), jp.ravel(x_max), jp.ravel(y_min), jp.ravel(y_max)


def build_x_bin_bounds(H, W, W_bin):
    """Build normalized horizontal bounds for each bin column.

    The left and right pixel centers of every bin column are mapped to the
    renderer's normalized image-plane x coordinate.

    # Arguments
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `W_bin`: Python int bin width in pixels.

    # Returns
    - `x_min`: JAX array with shape `(W // W_bin,)`.
    - `x_max`: JAX array with shape `(W // W_bin,)`.
    """
    base = jp.minimum(H, W)
    cols = jp.arange(W // W_bin)
    low = cols * W_bin + 0.5
    high = (cols + 1) * W_bin - 0.5
    return (low - W / 2.0) * 2.0 / base, (high - W / 2.0) * 2.0 / base


def build_y_bin_bounds(H, W, H_bin):
    """Build normalized vertical bounds for each bin row.

    Pixel rows increase downward, while normalized image-plane y increases
    upward. The returned `y_min` and `y_max` follow normalized coordinates.

    # Arguments
    - `H`: Python int image height in pixels.
    - `W`: Python int image width in pixels.
    - `H_bin`: Python int bin height in pixels.

    # Returns
    - `y_min`: JAX array with shape `(H // H_bin,)`.
    - `y_max`: JAX array with shape `(H // H_bin,)`.
    """
    base = jp.minimum(H, W)
    rows = jp.arange(H // H_bin)
    high = rows * H_bin + 0.5
    low = (rows + 1) * H_bin - 0.5
    return (H / 2.0 - low) * 2.0 / base, (H / 2.0 - high) * 2.0 / base


def bin_side(image_size, num_bins):
    """Compute the pixel side length of one bin along an image axis.

    # Arguments
    - `image_size`: Python int size of one image axis in pixels.
    - `num_bins`: Python int number of bins along that axis.

    # Returns
    - Python int bin side length in pixels.
    """
    return image_size // num_bins


def build_tile_pixel_coordinates(H, W, H_tiles, W_tiles, tile_arg):
    """Build normalized pixel centers for one tile or bin.

    The tile coordinate selects one regular tile from an `H_tiles` by
    `W_tiles` grid. Returned pixels are flattened in row-major order.

    # Arguments
    - `H`: Python int full image height in pixels.
    - `W`: Python int full image width in pixels.
    - `H_tiles`: Python int number of tiles along image height.
    - `W_tiles`: Python int number of tiles along image width.
    - `tile_arg`: JAX int array with shape `(2,)`, `(x_tile, y_tile)`.

    `Hb = H // H_tiles` and `Wb = W // W_tiles`.

    # Returns
    - JAX array with shape `(Hb * Wb, 2)`, normalized pixel centers.
    """
    W_tile_arg, H_tile_arg = tile_arg
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    H_start = tile_H * H_tile_arg
    W_start = tile_W * W_tile_arg
    cols = jp.arange(tile_W) + W_start + 0.5
    rows = jp.arange(tile_H) + H_start + 0.5
    return build_coordinates(H, W, rows, cols)


def build_coordinates(H, W, rows, cols):
    """Build normalized image-plane coordinates for pixel centers.

    Pixel columns and rows are converted to the same normalized coordinates
    as projected vertices. X increases rightward and y increases upward.

    # Arguments
    - `H`: Python int full image height in pixels.
    - `W`: Python int full image width in pixels.
    - `rows`: JAX array with shape `(R,)`, pixel center row coordinates.
    - `cols`: JAX array with shape `(C,)`, pixel center column coordinates.

    `R` is the number of selected rows and `C` is the number of selected
    columns.

    # Returns
    - JAX array with shape `(R * C, 2)`, normalized pixel centers.
    """
    col_grid, row_grid = jp.meshgrid(cols, rows)
    base = jp.minimum(H, W)
    x_grid = (col_grid - W / 2.0) * 2.0 / base
    y_grid = (H / 2.0 - row_grid) * 2.0 / base
    coords = [jp.ravel(x_grid), jp.ravel(y_grid)]
    return jp.stack(coords, axis=-1)


def compute_barycentric_coordinates(A, B, C, pixels):
    """Compute barycentric coordinates for pixels against many triangles.

    The signed triangle area normalizes edge-function weights. The result is
    face-major, so each face has barycentric weights for every pixel.

    # Arguments
    - `A`: JAX array with shape `(F, 2)`, first triangle vertex.
    - `B`: JAX array with shape `(F, 2)`, second triangle vertex.
    - `C`: JAX array with shape `(F, 2)`, third triangle vertex.
    - `pixels`: JAX array with shape `(P, 2)`, normalized pixel centers.

    `F` is the number of faces in the current chunk, and `P` is the number
    of pixels in the current bin.

    # Returns
    - `barycentric`: JAX array with shape `(F, P, 3)`.
    - `area`: JAX array with shape `(F,)`, signed twice-area.
    """
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
    """Clip barycentric coordinates to the triangle boundary or interior.

    Negative weights are set to zero and the remaining weights are
    renormalized. This gives a stable closest-on-triangle interpolation for
    depth when outside pixels are near the silhouette.

    # Arguments
    - `barycentric`: JAX array with shape `(..., 3)`.

    # Returns
    - JAX array with shape `(..., 3)`, non-negative weights summing to one.
    """
    barycentric = jp.maximum(barycentric, 0.0)
    scale = jp.maximum(jp.sum(barycentric, axis=-1, keepdims=True), EPSILON)
    return barycentric / scale


def compute_triangle_distance(A, B, C, pixels):
    """Compute squared distance from pixels to bounded triangle edges.

    The distance is the minimum squared distance to the three finite edge
    segments, not to infinite lines. Interior pixels also get their nearest
    edge distance here; the sign is added later by `compute_face_fragments`.

    # Arguments
    - `A`: JAX array with shape `(F, 2)`, first triangle vertex.
    - `B`: JAX array with shape `(F, 2)`, second triangle vertex.
    - `C`: JAX array with shape `(F, 2)`, third triangle vertex.
    - `pixels`: JAX array with shape `(P, 2)`, normalized pixel centers.

    `F` is the number of faces in the current chunk, and `P` is the number
    of pixels in the current bin.

    # Returns
    - JAX array with shape `(F, P)`, squared edge distance.
    """
    distance_AB = compute_line_distance(A, B, pixels)
    distance_BC = compute_line_distance(B, C, pixels)
    distance_CA = compute_line_distance(C, A, pixels)
    return jp.minimum(distance_AB, jp.minimum(distance_BC, distance_CA))


def compute_line_distance(start, end, pixels):
    """Compute squared distance from pixels to finite line segments.

    Pixels are projected onto each segment, the projection factor is clipped
    to `[0, 1]`, and the squared distance to that closest segment point is
    returned. Distances are orthogonal only when the unclipped projection lies
    inside the segment; near endpoints they measure endpoint distance.

    # Arguments
    - `start`: JAX array with shape `(F, 2)`, segment starts.
    - `end`: JAX array with shape `(F, 2)`, segment ends.
    - `pixels`: JAX array with shape `(P, 2)`, normalized pixel centers.

    `F` is the number of faces in the current chunk, and `P` is the number
    of pixels in the current bin.

    # Returns
    - JAX array with shape `(F, P)`, squared segment distance.
    """
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


def edge_function(point, start, end):
    """Compute signed 2D edge-function values.

    The edge function is the 2D cross product between `point - start` and
    `end - start`. It is positive or negative depending on which side of the
    directed edge the point lies.

    # Arguments
    - `point`: JAX array with trailing shape `(..., 2)`.
    - `start`: JAX array broadcastable to `point`, trailing shape `(..., 2)`.
    - `end`: JAX array broadcastable to `point`, trailing shape `(..., 2)`.

    # Returns
    - JAX array with broadcasted leading shape, containing signed areas.
    """
    return cross2D(point - start, end - start)


def cross2D(left, right):
    """Compute the scalar 2D cross product.

    The result is `left_x * right_y - left_y * right_x`, with normal JAX
    broadcasting over all leading dimensions.

    # Arguments
    - `left`: JAX array with trailing shape `(..., 2)`.
    - `right`: JAX array broadcastable to `left`, trailing shape `(..., 2)`.

    # Returns
    - JAX array with broadcasted leading shape.
    """
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
