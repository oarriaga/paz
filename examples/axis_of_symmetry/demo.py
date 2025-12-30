import numpy as np


def calculate_centroid(point_cloud):
    return np.mean(point_cloud, axis=0)


def center_point_cloud_data(point_cloud):
    centroid = calculate_centroid(point_cloud)
    return point_cloud - centroid, centroid


def compute_eigen_system(point_cloud):
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors


def sample_point_indices(total_point_count, sample_size):
    if total_point_count <= sample_size:
        return np.arange(total_point_count)
    return np.random.choice(total_point_count, sample_size, replace=False)


def reflect_points_across_vector(points, normal_vector):
    projections = np.dot(points, normal_vector)
    projection_vectors = np.outer(projections, normal_vector)
    return points - 2 * projection_vectors


def compute_alignment_error(test_points, reference_points):
    differences = (
        test_points[:, np.newaxis, :] - reference_points[np.newaxis, :, :]
    )
    distances = np.linalg.norm(differences, axis=2)
    minimum_distances = np.min(distances, axis=1)
    return np.mean(minimum_distances)


def evaluate_plane_normal(normal_vector, test_points, reference_points):
    reflected_points = reflect_points_across_vector(test_points, normal_vector)
    return compute_alignment_error(reflected_points, reference_points)


def determine_best_plane_normal(eigenvectors, test_points, reference_points):
    errors = []
    for index in range(3):
        current_error = evaluate_plane_normal(
            eigenvectors[:, index], test_points, reference_points
        )
        errors.append(current_error)
    return eigenvectors[:, np.argmin(errors)]


def calculate_axis_circularity_ratios(eigenvalues):
    ratio_0 = abs(1 - (eigenvalues[1] / eigenvalues[2]))
    ratio_1 = abs(1 - (eigenvalues[0] / eigenvalues[2]))
    ratio_2 = abs(1 - (eigenvalues[0] / eigenvalues[1]))
    return [ratio_0, ratio_1, ratio_2]


def identify_rotation_axis_vector(eigenvalues, eigenvectors):
    ratios = calculate_axis_circularity_ratios(eigenvalues)
    best_index = np.argmin(ratios)
    return eigenvectors[:, best_index]


def estimate_symmetry_plane(point_cloud, sample_size=1000):
    centered_points, centroid = center_point_cloud_data(point_cloud)
    eigenvalues, eigenvectors = compute_eigen_system(centered_points)
    sample_indices = sample_point_indices(point_cloud.shape[0], sample_size)

    sampled_points = centered_points[sample_indices]
    best_normal = determine_best_plane_normal(
        eigenvectors, sampled_points, sampled_points
    )

    return centroid, best_normal


def estimate_symmetry_axis(point_cloud):
    centered_points, centroid = center_point_cloud_data(point_cloud)
    eigenvalues, eigenvectors = compute_eigen_system(centered_points)
    axis_vector = identify_rotation_axis_vector(eigenvalues, eigenvectors)

    return centroid, axis_vector


# demo


def generate_cylinder_surface_points(radius, height, point_count):
    angles = np.random.uniform(0, 2 * np.pi, point_count)
    heights = np.random.uniform(-height / 2, height / 2, point_count)
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)
    return np.column_stack((x_coordinates, y_coordinates, heights))


def create_rotation_matrix_x(angle_radians):
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])


def create_rotation_matrix_z(angle_radians):
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def rotate_point_cloud(point_cloud, rotation_matrix):
    return np.dot(point_cloud, rotation_matrix.T)


def print_vector_comparison(label, estimated_vector, true_vector):
    dot_product = np.abs(np.dot(estimated_vector, true_vector))
    print(f"{label} Alignment (Dot Product): {dot_product:.4f}")
    print(f"  Estimated: {estimated_vector}")
    print(f"  True:      {true_vector}")


def execute_symmetry_test():
    total_points = 2000
    cylinder_radius = 1.0
    cylinder_height = 5.0

    local_cylinder = generate_cylinder_surface_points(
        cylinder_radius, cylinder_height, total_points
    )

    rotation_x = create_rotation_matrix_x(np.pi / 4)
    rotation_z = create_rotation_matrix_z(np.pi / 3)
    combined_rotation = np.dot(rotation_z, rotation_x)

    world_cylinder = rotate_point_cloud(local_cylinder, combined_rotation)

    estimated_center, estimated_plane_normal = estimate_symmetry_plane(
        world_cylinder
    )
    estimated_center_axis, estimated_axis_vector = estimate_symmetry_axis(
        world_cylinder
    )

    true_axis_vector = np.dot(combined_rotation, np.array([0, 0, 1]))

    print("--- Symmetry Detection Results ---")
    print_vector_comparison("Axis", estimated_axis_vector, true_axis_vector)

    is_orthogonal = (
        np.abs(np.dot(estimated_plane_normal, true_axis_vector)) < 0.1
    )
    print(f"\nPlane Normal is orthogonal to True Axis: {is_orthogonal}")
    print(f"  Plane Normal: {estimated_plane_normal}")


if __name__ == "__main__":
    execute_symmetry_test()
