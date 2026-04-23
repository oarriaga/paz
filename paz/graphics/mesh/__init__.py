from .types import Mesh, normalize_mesh_batch
from .intersect import (
    EPSILON,
    compute_f,
    intersect_mesh,
    intersect_canonical_mesh,
    intersect_chunked,
)
from .geometry import (
    build_edges,
    compute_canonical_normals,
    compute_normals,
    compute_normals_for_hits,
    compute_position,
    extract_points,
    transform_points,
)
from .patterns import (
    compute_mesh_base_colors,
    compute_mesh_pattern_colors_from_points,
    compute_mesh_pattern_colors_from_uv,
    compute_mesh_vertex_uvs,
    interpolate_triangle_values,
    interpolate_for_hits,
    compute_base_colors_for_hits,
)
from .shading import (
    compute_ambient,
    compute_base_color,
    compute_colors_for_hits,
    compute_diffuse,
    compute_mesh_colors,
    compute_specular,
    vertex_colors_to_face_colors,
)
from .render import (
    mask_out_mesh,
    postprocess,
    postprocess_depth,
    render,
    render_depth,
    render_masks,
    render_mesh,
    render_mesh_depth,
    select_closest_color,
    to_color_image,
    to_depth_image,
)
from .tile import (
    assemble,
    assert_exact_tile_side,
    build_tile_rays,
    make_ray_origins,
    make_ray_targets,
    make_tile_coordinates,
    render_depth_tile,
    render_tile,
    tile_render,
    tile_render_depth,
    tile_render_masks,
    transform_tile_rays,
)
from .silhouette import (
    BinArgs,
    count_binned_faces,
    tile_render_binned_soft_mask,
)
from .builders import (
    build_cube,
    build_sphere,
    fill_bottom_with_last,
    fill_mesh,
    load_mesh,
    merge_meshes,
)
