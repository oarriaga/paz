import jax.numpy as jp
import jax
import paz


def compute_edge_normals(vertices, normal):
    r"""Computes \hat{e}_i = normalize((v_{i+1} - v_{i}) \times \hat{n})

          ^
       ___|___
       | face |
    <--|  o +z|--> edge normals
       |normal|
       -------
          |
          v
    """
    rolled_vertices = jp.roll(vertices, 1, axis=0)
    edges = vertices - rolled_vertices
    edge_normals = jax.vmap(jp.cross, in_axes=[0, None])(edges, normal)
    norm = jp.linalg.norm(edge_normals, axis=1, keepdims=True)
    safe_norm = jp.where(norm == 0, 1.0, norm)  # TODO
    edge_normals = edge_normals / safe_norm
    return rolled_vertices, edge_normals


def intersect_plane(origin, direction, plane_point, plane_normal):
    """Given a ray:        r(t) = origin + t · direction
    and a plane:           plane_normal · (x - point) = 0
    find t lying on plane: plane_normal · (point + t · direction - point) = 0
    """
    numerator = jp.dot(plane_normal, plane_point - origin)
    denominator = jp.dot(plane_normal, direction)
    t = numerator / (denominator + 1e-9)
    return origin + t * direction


def project_segment_onto_plane(point_A, point_B, plane_point, plane_normal):
    """Gets the closest point between a line segment and a plane."""
    # Given a line segment parametrized as l(t) = a + t * (b - a).
    # Plug into the plane equation (check intersection) dot(n, l(t)) - d = 0.
    # We get the plane equation as dot(n, a) + t * dot(n, b - a) - d = 0.
    # Solving for t = (d - dot(n, a)) / dot(n, b - a).
    # Clip t to be in [0, 1] to be on the line segment.
    direction_A_to_B = point_B - point_A
    distance = jp.sum(plane_point * plane_normal)
    denominator = jp.sum(plane_normal * direction_A_to_B)
    safe_denominator = denominator + (1e-6 * (denominator == 0.0))
    t = (distance - jp.sum(plane_normal * point_A)) / safe_denominator
    segment_point = point_A + (jp.clip(t, 0, 1) * direction_A_to_B)
    return segment_point


def project_to_polygon(vertices_A, normal_A, vertices_B, normal_B):
    plane_point, plane_normal, direction = vertices_B[0], normal_B, normal_A
    project = paz.lock(intersect_plane, direction, plane_point, plane_normal)
    return jax.vmap(project)(vertices_A)


def clip_edge_point(point_0, point_1, is_point_0_in_front, clipped_points):

    @jax.vmap
    def choose_edge_points(is_in_front, clip_point):
        return jp.where(is_in_front, clip_point, point_0)

    new_edge_points = choose_edge_points(is_point_0_in_front, clipped_points)
    # Pick the clipped point that is most along the edge direction.
    # This degenerates to picking the original point p0 if p0 is *not* in front of any clipping planes.  # fmt: skip
    distances = jp.dot(new_edge_points - point_0, point_1 - point_0)
    new_point = new_edge_points[jp.argmax(distances)]
    return new_point


def clip_edges_against_planes(point_A0, point_A1, plane_points, plane_normals):
    """Clips an edge against side planes."""
    epsilon = 1e-6
    vdot = jax.vmap(jp.dot)
    A0_in_front = vdot(point_A0 - plane_points, plane_normals) > epsilon
    A1_in_front = vdot(point_A1 - plane_points, plane_normals) > epsilon

    project_to_planes = jax.vmap(project_segment_onto_plane, [None, None, 0, 0])
    plane = (plane_points, plane_normals)
    clipped_points = project_to_planes(point_A0, point_A1, *plane)
    A0_new = clip_edge_point(point_A0, point_A1, A0_in_front, clipped_points)
    A1_new = clip_edge_point(point_A1, point_A0, A1_in_front, clipped_points)
    clipped_points = jp.array([A0_new, A1_new])

    # Keep the original points if both points are in front of any of the clipping planes, rather than creating a new clipped edge.  # fmt: skip
    # If the entire subject edge is in front of any clipping plane, we need to grab an edge from the clipping polygon instead.      # fmt: skip
    mask = jp.logical_not(jp.any(A0_in_front & A1_in_front))
    new_points = jp.where(mask, clipped_points, jp.array([point_A0, point_A1]))
    # Mask out crossing clipped edge points.
    mask = jp.where((point_A0 - point_A1).dot(new_points[0] - new_points[1]) < 0, False, mask)  # fmt: skip
    return new_points, jp.array([mask, mask])
