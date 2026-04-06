import jax.numpy as jp


def build_laplacian(vertices, faces):
    """Build the dense mesh Laplacian L = D - A.

    Each triangle [a, b, c] adds the undirected edges
    (a, b), (b, c), and (c, a). Adjacent vertices get -1
    and each diagonal entry counts how many neighbors that
    vertex has.
    """

    def build_triangle_edges(faces):
        edge_starts = faces[:, [0, 1, 2]].flatten()
        edge_ends = faces[:, [1, 2, 0]].flatten()
        return jp.stack((edge_starts, edge_ends))

    def make_undirected(edges):
        edge_starts = edges[0]
        edge_ends = edges[1]
        rows = jp.concatenate((edge_starts, edge_ends))
        cols = jp.concatenate((edge_ends, edge_starts))
        return jp.unique(jp.stack((rows, cols)), axis=1)

    def build_diagonal_indices(edges):
        rows = edges[0]
        return jp.stack((rows, rows))

    triangle_edges = build_triangle_edges(faces)
    undirected_edges = make_undirected(triangle_edges)
    diagonal_indices = build_diagonal_indices(undirected_edges)
    stacked_indices = (undirected_edges, diagonal_indices)
    laplacian_indices = jp.concatenate(stacked_indices, axis=1)
    num_edges = undirected_edges.shape[1]
    edge_values = -jp.ones(num_edges)
    degree_values = jp.ones(num_edges)
    laplacian_values = jp.concatenate((edge_values, degree_values))
    laplacian = jp.zeros((len(vertices), len(vertices)))
    row_indices = laplacian_indices[0]
    col_indices = laplacian_indices[1]
    entries = zip(row_indices, col_indices, laplacian_values)
    for row, col, value in entries:
        prev = laplacian[row, col]
        laplacian = laplacian.at[row, col].set(jp.add(prev, value))
    return laplacian


def compute_volume(vertices, faces):

    def extract_face_vertices(vertices, faces):
        point_a = vertices[faces[:, 0]]
        point_b = vertices[faces[:, 1]]
        point_c = vertices[faces[:, 2]]
        return point_a, point_b, point_c

    point_a, point_b, point_c = extract_face_vertices(vertices, faces)
    cross_products = jp.cross(point_b, point_c)
    signed_terms = jp.sum(point_a * cross_products, axis=1)
    signed_volume = signed_terms.sum() / 6.0
    return jp.abs(signed_volume)
