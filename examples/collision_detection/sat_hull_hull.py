import jax.numpy as jp
import jax
import paz


def sat_hull_hull(
    faces_A,
    faces_B,
    vertices_A,
    vertices_B,
    normals_A,
    normals_B,
    unique_edges_A,
    unique_edges_B,
):
    """Runs the Separating Axis Test for a pair of hulls."""
    # Given two convex hulls, the Separating Axis Test finds a separating axis
    # between all edge pairs and face pairs. Edge pairs create a single contact
    # point and face pairs create a contact manifold (up to four contact points)
    # We return both the edge and face contacts. Valid contacts can be checked
    # with penetration > 0. Resulting edge contacts should be preferred over
    # face contacts.
    axes = get_separating_axes(
        unique_edges_A, unique_edges_B, normals_A, normals_B
    )
    best_axis, best_sign, best_index = choose_best_axis(
        axes, vertices_A, vertices_B
    )
    is_edge_contact = best_index >= (normals_A.shape[0] + normals_B.shape[0])
    position, normal, penetration = results_face_contact(
        faces_A, faces_B, normals_A, normals_B, best_axis, best_sign
    )
    position, penetration = results_edge_contact(
        is_edge_contact, position, penetration
    )
    return position, normal, penetration


def get_separating_axes(edges_A, edges_B, normals_A, normals_B):
    edge_direction_A = edges_A[:, 0] - edges_A[:, 1]
    edge_direction_B = edges_B[:, 0] - edges_B[:, 1]
    edge_directions_A = jp.tile(edge_direction_A, len(edges_B), 1)
    edge_directions_B = jp.repeat(edge_direction_B, len(edges_A), axis=0)
    edge_edge_axes = jax.vmap(jp.cross)(edge_directions_A, edge_directions_B)
    edge_edge_axes = jax.vmap(paz.lock(paz.algebra.normalize, 0))(
        edge_edge_axes
    )
    axes = jp.concatenate([normals_A, normals_B, edge_edge_axes])
    return axes
