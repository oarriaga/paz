import jax
import jax.numpy as jp
import paz
import pytest

from paz.backend.pointcloud import (
    chamfer_distance,
    centroid_distance,
    color,
    compute_approx_emd,
    compute_bounding_box,
    compute_bounding_volume,
    compute_centroid,
    compute_fscore,
    compute_nearest_squared_distances,
    compute_pairwise_squared_distances,
    compute_precision_recall_fscore,
    filter_above_plane,
    from_depth,
    hausdorff_distance,
    mask,
    mean,
    mean_nearest_neighbor_distance,
    merge,
    move_along_normals,
    remove_outliers,
    sample,
    split,
    stdv,
    to_camera_coordinates,
    to_depth,
    transform,
    bound,
)


def build_intrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0):
    return jp.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def test_split_returns_components():
    pointcloud = jp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x, y, z = split(pointcloud)
    assert jp.allclose(x, jp.array([1.0, 4.0]))
    assert jp.allclose(y, jp.array([2.0, 5.0]))
    assert jp.allclose(z, jp.array([3.0, 6.0]))


def test_split_preserves_order():
    pointcloud = jp.array([[3.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    x, _, _ = split(pointcloud)
    assert jp.allclose(x, jp.array([3.0, 2.0]))


def test_bound_filters_zero_and_far():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    filtered = bound(pointcloud, 0.8)
    assert jp.allclose(filtered, jp.array([[0.0, 0.0, 0.5]]))


def test_bound_keeps_near_points():
    pointcloud = jp.array([[0.1, 0.0, 0.2], [0.2, 0.0, 0.3]])
    filtered = bound(pointcloud, 1.0)
    assert filtered.shape[0] == 2


def test_merge_two_pointclouds():
    pointcloud_a = jp.array([[0.0, 0.0, 0.0]])
    pointcloud_b = jp.array([[1.0, 0.0, 0.0]])
    merged = merge([pointcloud_a, pointcloud_b])
    assert merged.shape == (2, 3)


def test_merge_three_pointclouds():
    pointclouds = [jp.zeros((1, 3)), jp.ones((1, 3)), jp.full((1, 3), 2.0)]
    merged = merge(pointclouds)
    assert merged.shape[0] == 3


def test_transform_identity():
    pointcloud = jp.array([[1.0, 2.0, 3.0]])
    identity = jp.eye(4)
    transformed = transform(pointcloud, identity)
    assert jp.allclose(transformed, pointcloud)


def test_transform_translation():
    pointcloud = jp.array([[0.0, 0.0, 0.0]])
    translation = paz.SE3.translation(jp.array([1.0, 2.0, 3.0]))
    transformed = transform(pointcloud, translation)
    assert jp.allclose(transformed, jp.array([[1.0, 2.0, 3.0]]))


def test_color_default_green():
    pointcloud = jp.zeros((2, 3))
    colors = color(pointcloud)
    assert jp.allclose(colors, jp.array([[0, 255, 0], [0, 255, 0]]))


def test_color_custom_color():
    pointcloud = jp.zeros((1, 3))
    colors = color(pointcloud, color=[10, 20, 30])
    assert jp.allclose(colors, jp.array([[10, 20, 30]]))


def test_mean_returns_average():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = mean(pointcloud)
    assert jp.allclose(result, jp.array([1.0, 0.0, 0.0]))


def test_mean_handles_negative():
    pointcloud = jp.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    result = mean(pointcloud)
    assert jp.allclose(result, jp.array([0.0, 0.0, 0.0]))


def test_stdv_zero_for_identical():
    pointcloud = jp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    result = stdv(pointcloud)
    assert jp.allclose(result, jp.zeros(3))


def test_stdv_simple():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = stdv(pointcloud)
    assert float(result[0]) == pytest.approx(1.0)


def test_mask_applies_shape_mask():
    pointcloud = jp.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    shape_mask = jp.array([[1], [0]])
    filtered = mask(pointcloud, shape_mask, max_depth=1.0)
    assert jp.allclose(filtered, jp.array([[0.0, 0.0, 0.5]]))


def test_mask_applies_depth_and_zero():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    shape_mask = jp.array([[1], [1]])
    filtered = mask(pointcloud, shape_mask, max_depth=1.0)
    assert filtered.shape[0] == 0


def test_from_depth_single_pixel():
    depth = jp.array([[1.0]])
    intrinsics = build_intrinsics()
    pointcloud = from_depth(depth, intrinsics)
    assert jp.allclose(pointcloud, jp.array([[0.0, 0.0, 1.0]]))


def test_from_depth_two_pixels():
    depth = jp.array([[1.0, 2.0]])
    intrinsics = build_intrinsics()
    pointcloud = from_depth(depth, intrinsics)
    expected = jp.array([[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    assert jp.allclose(pointcloud, expected)


def test_to_depth_raises_typeerror_default():
    intrinsics = build_intrinsics()
    pointcloud = jp.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    with pytest.raises(TypeError):
        _ = to_depth(pointcloud, intrinsics, H=1, W=1)


def test_to_depth_raises_typeerror_with_min_depth():
    intrinsics = build_intrinsics()
    pointcloud = jp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    with pytest.raises(TypeError):
        _ = to_depth(pointcloud, intrinsics, H=1, W=1, min_depth=1.5)


def test_sample_returns_requested_count():
    key = jax.random.PRNGKey(0)
    pointcloud = jp.arange(12.0).reshape(4, 3)
    sampled = sample(key, pointcloud, num_points=2)
    assert sampled.shape[0] == 2


def test_sample_deterministic_for_same_key():
    key = jax.random.PRNGKey(0)
    pointcloud = jp.arange(12.0).reshape(4, 3)
    sample_a = sample(key, pointcloud, num_points=2)
    sample_b = sample(key, pointcloud, num_points=2)
    assert jp.allclose(sample_a, sample_b)


def test_to_camera_coordinates_identity():
    pointcloud = jp.array([[1.0, 2.0, 1.0]])
    intrinsics = build_intrinsics()
    coords = to_camera_coordinates(pointcloud, intrinsics)
    assert jp.allclose(coords, jp.array([[1.0, 2.0]]))


def test_to_camera_coordinates_offset():
    pointcloud = jp.array([[1.0, 1.0, 1.0]])
    intrinsics = build_intrinsics(cx=0.5, cy=0.5)
    coords = to_camera_coordinates(pointcloud, intrinsics)
    assert jp.allclose(coords, jp.array([[1.5, 1.5]]))


def test_compute_bounding_box_simple():
    pointcloud = jp.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    bounds = compute_bounding_box(pointcloud)
    assert bounds == (0.0, 1.0, 2.0, 2.0, 3.0, 4.0)


def test_compute_bounding_box_negative():
    pointcloud = jp.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    bounds = compute_bounding_box(pointcloud)
    assert bounds == (-1.0, -2.0, -3.0, 1.0, 2.0, 3.0)


def test_remove_outliers_returns_empty_for_outliers():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    filtered = remove_outliers(pointcloud)
    assert filtered.shape[0] == 0


def test_remove_outliers_returns_empty_for_small_stdv():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    filtered = remove_outliers(pointcloud, num_stdvs=0.1)
    assert filtered.shape[0] == 0


def test_compute_bounding_volume_cube():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    volume = compute_bounding_volume(pointcloud)
    assert float(volume) == pytest.approx(1.0)


def test_compute_bounding_volume_flat():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    volume = compute_bounding_volume(pointcloud)
    assert float(volume) == pytest.approx(0.0)


def test_move_along_normals_positive():
    pointcloud = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    moved = move_along_normals(pointcloud, normals, distance=2.0)
    assert jp.allclose(moved, jp.array([[0.0, 2.0, 0.0]]))


def test_move_along_normals_negative():
    pointcloud = jp.array([[0.0, 0.0, 0.0]])
    normals = jp.array([[1.0, 0.0, 0.0]])
    moved = move_along_normals(pointcloud, normals, distance=-1.0)
    assert jp.allclose(moved, jp.array([[-1.0, 0.0, 0.0]]))


def test_filter_above_plane_identity():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    plane_to_world = jp.eye(4)
    filtered = filter_above_plane(pointcloud, plane_to_world, min_height=0.05)
    assert jp.allclose(filtered, jp.array([[0.0, 0.1, 0.0]]))


def test_filter_above_plane_zero_threshold():
    pointcloud = jp.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    plane_to_world = jp.eye(4)
    filtered = filter_above_plane(pointcloud, plane_to_world, min_height=0.0)
    assert filtered.shape[0] == 1


def test_compute_pairwise_squared_distances_simple():
    points_a = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points_b = jp.array([[0.0, 0.0, 0.0]])
    distances = compute_pairwise_squared_distances(points_a, points_b)
    expected = jp.array([[0.0], [1.0]])
    assert jp.allclose(distances, expected)


def test_compute_pairwise_squared_distances_symmetry():
    points_a = jp.array([[0.0, 0.0, 0.0]])
    points_b = jp.array([[2.0, 0.0, 0.0]])
    distances = compute_pairwise_squared_distances(points_a, points_b)
    assert jp.allclose(distances, jp.array([[4.0]]))


def test_compute_nearest_squared_distances_different_sizes():
    points_a = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points_b = jp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    nearest = compute_nearest_squared_distances(points_a, points_b)
    expected = jp.array([1.0, 1.0])
    assert jp.allclose(nearest, expected)


def test_compute_nearest_squared_distances_identical():
    points = jp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    nearest = compute_nearest_squared_distances(points, points)
    assert jp.allclose(nearest, jp.zeros(2))


def test_compute_centroid_returns_mean():
    points = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    centroid = compute_centroid(points)
    expected = jp.array([1.0, 0.0, 0.0])
    assert jp.allclose(centroid, expected)


def test_compute_centroid_offset():
    points = jp.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    centroid = compute_centroid(points)
    assert jp.allclose(centroid, jp.array([3.0, 0.0, 0.0]))


def test_centroid_distance_matches_expected():
    points_a = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points_b = jp.array([[3.0, 0.0, 0.0]])
    distance = centroid_distance(points_a, points_b)
    assert float(distance) == pytest.approx(2.0)


def test_centroid_distance_zero_for_identical():
    points = jp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    distance = centroid_distance(points, points)
    assert float(distance) == pytest.approx(0.0)


def test_mean_nearest_neighbor_distance_simple():
    points_a = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points_b = jp.array([[1.0, 0.0, 0.0]])
    distance = mean_nearest_neighbor_distance(points_a, points_b)
    assert float(distance) == pytest.approx(1.0)


def test_mean_nearest_neighbor_distance_identical():
    points = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distance = mean_nearest_neighbor_distance(points, points)
    assert float(distance) == pytest.approx(0.0)


def test_chamfer_distance_zero_for_identical():
    points = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distance = chamfer_distance(points, points)
    assert float(distance) == pytest.approx(0.0)


def test_chamfer_distance_collinear_points():
    points_a = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points_b = jp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    distance = chamfer_distance(points_a, points_b)
    assert float(distance) == pytest.approx(2.0)


def test_hausdorff_distance_expected_value():
    points_a = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points_b = jp.array([[1.0, 0.0, 0.0]])
    distance = hausdorff_distance(points_a, points_b)
    assert float(distance) == pytest.approx(1.0)


def test_hausdorff_distance_with_duplicate_points():
    points_a = jp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    points_b = jp.array([[1.0, 0.0, 0.0]])
    distance = hausdorff_distance(points_a, points_b)
    assert float(distance) == pytest.approx(1.0)


def test_compute_precision_recall_fscore_threshold():
    points_a = jp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    points_b = jp.array([[0.05, 0.0, 0.0]])
    metrics = compute_precision_recall_fscore(points_a, points_b, 0.1)
    assert float(metrics.precision) == pytest.approx(1.0)
    assert float(metrics.recall) == pytest.approx(0.5)
    assert float(metrics.f_score) == pytest.approx(2.0 / 3.0)


def test_compute_precision_recall_fscore_perfect():
    points = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    metrics = compute_precision_recall_fscore(points, points, 0.1)
    assert float(metrics.f_score) == pytest.approx(1.0)


def test_compute_fscore_matches_metrics():
    points_a = jp.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    points_b = jp.array([[0.05, 0.0, 0.0]])
    f_score = compute_fscore(points_a, points_b, 0.1)
    assert float(f_score) == pytest.approx(2.0 / 3.0)


def test_compute_fscore_perfect():
    points = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    f_score = compute_fscore(points, points, 0.1)
    assert float(f_score) == pytest.approx(1.0)


def test_compute_approx_emd_different_sizes():
    points_a = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points_b = jp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    distance = compute_approx_emd(points_a, points_b)
    assert float(distance) == pytest.approx(0.5)


def test_compute_approx_emd_identical():
    points = jp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distance = compute_approx_emd(points, points)
    assert float(distance) == pytest.approx(0.0)
