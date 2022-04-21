import pytest
import os
import numpy as np
from tensorflow.keras.utils import get_file
from paz.abstract import Box2D, Pose6D

from paz.backend.image import load_image
from paz.backend.camera import Camera
from paz.pipelines import PIX2YCBTools6D


@pytest.fixture
def true_boxes2D():
    boxes = [
        Box2D(np.array([691, 324, 850, 510]), 0.9741702079, "037_scissors"),
        Box2D(np.array([326, 471, 550, 679]), 0.8022266626, "035_power_drill"),
        Box2D(np.array([436, 453, 706, 679]), 0.8869557976, "051_large_clamp"),
        Box2D(np.array([210, 469, 428, 679]), 0.7902939319, "051_large_clamp")]
    return boxes


@pytest.fixture
def true_poses6D():
    quaternion = [-0.04797872994777336, -0.21049879282908107,
                  0.937455586206986, 0.27306651859833425]
    translation = [0.5038849913659514, 0.09749278429483288, 1.1208797474903172]
    class_name = '037_scissors'
    pose6D_1 = Pose6D(quaternion, translation, class_name)

    quaternion = [-0.9554521536729669, 0.01142031139518192,
                  -0.2756557125195054, 0.10485555152060512]
    translation = [0.03445023852822503, 0.3832470770078622, 1.3231460303725164]
    class_name = '035_power_drill'
    pose6D_2 = Pose6D(quaternion, translation, class_name)

    quaternion = [0.7504569098767596, 0.5992670685201711,
                  0.15718646021698382, 0.2301864977141796]
    translation = [0.24124956284242208, 0.38640316245506734, 1.27936438317585]
    class_name = '051_large_clamp'
    pose6D_3 = Pose6D(quaternion, translation, class_name)

    quaternion = [0.7472538632802124, 0.5179774273151441,
                  0.400522930697268, 0.11354483955888982]
    translation = [-0.15736611266881423, 0.3458493586769305, 1.178571841067687]
    class_name = '051_large_clamp'
    pose6D_4 = Pose6D(quaternion, translation, class_name)

    return [pose6D_1, pose6D_2, pose6D_3, pose6D_4]


@pytest.fixture
def image_with_YCB_objects():
    URL = ('https://github.com/oarriaga/altamira-data/releases/download'
           '/v0.9.1/image_with_YCB_objects.jpg')
    filename = os.path.basename(URL)
    fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    image = load_image(fullpath)
    return image


def assert_boxes2D(true_boxes2D, pred_boxes2D):
    assert len(pred_boxes2D) == len(true_boxes2D)
    for true_box2D, pred_box2D in zip(true_boxes2D, pred_boxes2D):
        assert np.allclose(true_box2D.coordinates, pred_box2D.coordinates)
        assert np.allclose(true_box2D.score, pred_box2D.score)
        assert (true_box2D.class_name == pred_box2D.class_name)


def assert_poses6D(true_poses6D, pred_poses6D):
    assert len(pred_poses6D) == len(true_poses6D)
    for true_pose6D, true_pose6D in zip(true_poses6D, pred_poses6D):
        assert np.allclose(true_pose6D.quaternion, true_pose6D.quaternion)
        assert np.allclose(true_pose6D.translation, true_pose6D.translation)
        assert (true_pose6D.class_name == true_pose6D.class_name)


def test_PIX2YCBTools6D(image_with_YCB_objects, true_boxes2D, true_poses6D):
    camera = Camera()
    camera.intrinsics_from_HFOV(55, image_with_YCB_objects.shape)
    pipeline = PIX2YCBTools6D(camera)
    inferences = pipeline(image_with_YCB_objects)
    assert_boxes2D(true_boxes2D, inferences['boxes2D'])
    assert_poses6D(true_poses6D, inferences['poses6D'])
