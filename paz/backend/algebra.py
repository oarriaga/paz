import jax.numpy as jp


def add_one(point):
    return jp.concatenate([point, jp.ones(1)]).reshape(-1, 1)


def add_ones(points):
    ones = jp.ones((len(points), 1))
    points = jp.concatenate([points, ones], axis=-1)
    return points


def transform_points(affine_transform, points):
    points = add_ones(points)
    points = jp.matmul(affine_transform, points.T).T
    return points[:, :3]


def dehomogenize_coordinates(homogenous_point):
    homogenous_point = jp.squeeze(homogenous_point, axis=1)
    u, v, w = homogenous_point
    return jp.array([u / w, v / w])
