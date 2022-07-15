from ..abstract import Processor
from ..backend.groups import rotation_vector_to_quaternion
from ..backend.groups import rotation_vector_to_rotation_matrix
from ..backend.groups import to_affine_matrix


class RotationVectorToQuaternion(Processor):
    """Transforms rotation vector into quaternion.
    """
    def __init__(self):
        super(RotationVectorToQuaternion, self).__init__()

    def call(self, rotation_vector):
        quaternion = rotation_vector_to_quaternion(rotation_vector)
        return quaternion


class RotationVectorToRotationMatrix(Processor):
    """Transforms rotation vector into a rotation matrix.
    """
    def __init__(self):
        super(RotationVectorToRotationMatrix, self).__init__()

    def call(self, rotation_vector):
        return rotation_vector_to_rotation_matrix(rotation_vector)


class ToAffineMatrix(Processor):
    """Builds affine matrix from a rotation matrix and a translation vector.
    """
    def __init__(self):
        super(ToAffineMatrix, self).__init__()

    def call(self, rotation_matrix, translation):
        affine_matrix = to_affine_matrix(rotation_matrix, translation)
        return affine_matrix
