import numpy as np
import tensorflow.keras.backend as K


class Projector(object):
    """Projects keypoints from image coordinates to 3D space and viceversa.
    This model uses the camera focal length and the depth estimation of a point
    to project it to image coordinates. It works with numpy matrices or
    tensorflow values. See ``use_numpy``.

    # Arguments
        focal_length: Float. Focal length of camera used to generate keypoints.
        use_numpy: Boolean. If `True` both unproject and project functions
            take numpy arrays as inputs. If `False` takes tf.tensors as inputs.
    """
    def __init__(self, focal_length, use_numpy=False):
        self.focal_length = focal_length
        self.project = self._project_keras
        self.unproject = self._unproject_keras
        if use_numpy:
            self.project = self._project_numpy
            self.unproject = self._unproject_numpy

    def _project_keras(self, xyzw):
        z = xyzw[:, :, 2:3] + 1e-8
        x = - (self.focal_length / z) * xyzw[:, :, 0:1]
        y = - (self.focal_length / z) * xyzw[:, :, 1:2]
        return K.concatenate([x, y, z], axis=2)

    def _project_numpy(self, xyzw):
        z = xyzw[:, :, 2:3] + 1e-8
        x = - (self.focal_length / z) * xyzw[:, :, 0:1]
        y = - (self.focal_length / z) * xyzw[:, :, 1:2]
        return np.concatenate([x, y, z], axis=2)

    def _unproject_keras(self, xyz):
        z = xyz[:, :, 2:3]
        x = - (z / self.focal_length) * xyz[:, :, 0:1]
        y = - (z / self.focal_length) * xyz[:, :, 1:2]
        w = K.ones_like(z)
        xyzw = K.concatenate([x, y, z, w], axis=2)
        return xyzw

    def _unproject_numpy(self, xyz):
        z = xyz[:, :, 2:3]
        x = - (z / self.focal_length) * xyz[:, :, 0:1]
        y = - (z / self.focal_length) * xyz[:, :, 1:2]
        w = np.ones_like(z)
        xyzw = np.concatenate([x, y, z, w], axis=2)
        return xyzw
