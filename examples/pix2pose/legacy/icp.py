import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_affine_matrix(pointcloud_A, pointcloud_B):
    '''Calculates affine transform with the best least-squares fit transforming
        keypoints A to keypoints B.

    # Argument:
        pointcloud_A: Array of shape (num_keypoints, 3).
        pointcloud_B: Array of shape (num_keypoints, 3).

    # Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
    '''
    # assert pointcloud_A.shape == pointcloud_B.shape
    # translate points to their centroids
    centroid3D_A = np.mean(pointcloud_A, axis=0)
    centroid3D_B = np.mean(pointcloud_B, axis=0)
    centered_keypoints3D_A = pointcloud_A - centroid3D_A
    centered_keypoints3D_B = pointcloud_B - centroid3D_B

    covariance = np.dot(centered_keypoints3D_A.T, centered_keypoints3D_B)
    U, S, Vt = np.linalg.svd(covariance)
    # compute rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # resolve special reflection case
    if np.linalg.det(rotation_matrix) < 0:
        Vt[3 - 1, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # compute translation
    translation3D = centroid3D_B.T - np.dot(rotation_matrix, centroid3D_A.T)
    print(translation3D.shape)
    affine_matrix = to_affine_matrix(rotation_matrix, translation3D)
    return affine_matrix


def to_affine_matrix(rotation_matrix, translation_vector):
    translation_vector = translation_vector.reshape(3, 1)
    affine = np.concatenate([rotation_matrix, translation_vector], axis=1)
    affine = np.concatenate([affine, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    return affine


def nearest_neighbor(pointcloud_A, pointcloud_B):
    '''Find the nearest (Euclidean) neighbor in dst for each point in src
    # Arguments:
        src: Nxm array of points
        dst: Nxm array of points
    # Returns:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # assert pointcloud_A.shape == pointcloud_B.shape
    model = NearestNeighbors(n_neighbors=1)
    model.fit(pointcloud_B)
    distances, indices = model.kneighbors(pointcloud_A, return_distance=True)
    return distances.ravel(), indices.ravel()


def add_homogenous_coordinate(keypoints3D):
    num_keypoints = len(keypoints3D)
    # ones = np.ones_like(num_keypoints).reshape(-1, 1)
    ones = np.ones(num_keypoints).reshape(-1, 1)
    homogenous_keypoints3D = np.concatenate([keypoints3D, ones], axis=1)
    return homogenous_keypoints3D


def iterative_closes_point(pointcloud_A, pointcloud_B, initial_pose=None,
                           max_iterations=20, tolerance=1e-3):
    '''Find best least square fit that transforms pointcloud A to pointcloud B.
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        initial_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    # assert pointcloud_A.shape == pointcloud_B.shape
    pointcloud_A = add_homogenous_coordinate(pointcloud_A)
    pointcloud_B = add_homogenous_coordinate(pointcloud_B)
    pointcloud_A_0 = np.copy(pointcloud_A)
    if initial_pose is not None:
        pointcloud_A = np.dot(initial_pose, pointcloud_A.T).T
    previous_error = 0
    for iteration_arg in range(max_iterations):
        distances, indices = nearest_neighbor(pointcloud_A, pointcloud_B)
        print(indices.shape, pointcloud_A.shape, pointcloud_B.shape)
        pointcloud_B = pointcloud_B[indices]
        print(pointcloud_B.shape)
        print('***********************')
        affine_matrix = calculate_affine_matrix(pointcloud_A[:, :3], pointcloud_B[:, :3])
        pointcloud_A = np.dot(affine_matrix, pointcloud_A.T).T
        mean_error = np.mean(distances)
        print(mean_error)
        if np.abs(previous_error - mean_error) < tolerance:
            break
        previous_error = mean_error
    affine_transform = calculate_affine_matrix(pointcloud_A_0[:, :3], pointcloud_A[:, :3])
    return affine_transform, distances, iteration_arg
