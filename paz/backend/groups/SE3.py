import numpy as np


def to_affine_matrix(rotation_matrix, translation):
    """Builds affine matrix from rotation matrix and translation vector.

    # Arguments
        rotation_matrix: Array (3, 3). Representing a rotation matrix.
        translation: Array (3). Translation vector.

    # Returns
        Array (4, 4) representing an affine matrix.
    """
    if len(translation) != 3:
        raise ValueError('Translation should be of lenght 3')
    if rotation_matrix.shape != (3, 3):
        raise ValueError('Rotation matrix should be of shape (3, 3)')
    translation = translation.reshape(3, 1)
    affine_top = np.concatenate([rotation_matrix, translation], axis=1)
    affine_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = np.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def to_affine_matrices(rotations, translations):
    """Construct affine matrices for rotation matrices vector and
       translation vector.

    # Arguments
        ratations: Rotation matrix vector [N, 3, 3].
        translations: Translation vector [N, 3].

    # Returns
        Transformation matrix [N, 4, 4]
    """
    affine_matrices = []
    for rotation, translation in zip(rotations, translations):
        transformation = to_affine_matrix(rotation, translation)
        affine_matrices.append(transformation)
    return np.array(affine_matrices)
