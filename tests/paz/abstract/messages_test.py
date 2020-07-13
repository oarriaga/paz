import numpy as np
from paz.abstract.messages import Box2D, Pose6D, Keypoint3D

def test_Box2D():
    """ Unit test for Bounding box 2D with class
    and score
    """
    coordinates = [219, 49, 319, 179]
    score = 10.0
    point = [250, 60]
    Box2D_class = Box2D(coordinates, score)
    assert (Box2D_class.contains(point) == True)

def test_Pose6D():
    """Unit test for Pose estimation 
    """
    quaternion = np.array([-0.4732069, 0.5253096, 0.4732069, 0.5255476])
    translation = np.array([1.0, 0.765, 0])
    rotation_vector = np.array([1., -0.994522, 0.104528])
    Pose6D_class = Pose6D(quaternion, translation)
    result = Pose6D_class.from_rotation_vector(rotation_vector, translation)
    assert(result.translation.all() == translation.all())

test_Box2D()
test_Pose6D()