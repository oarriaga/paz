from paz.graphics.types import Mesh
from .intersect import (
    EPSILON,
    compute_f,
    intersect_mesh,
    intersect_triangles,
    intersect_canonical_mesh,
    intersect_chunked,
)
from .geometry import (
    build_edges,
    compute_canonical_normals,
    compute_normals,
    compute_normals_for_hits,
    compute_triangle_normals,
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
from .render import (
    render_coordinates,
)
from .tile import (
    assemble,
    assert_exact_tile_side,
    build_tile_rays,
    make_ray_origins,
    make_ray_targets,
    make_tile_coordinates,
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
    load_mesh,
)
