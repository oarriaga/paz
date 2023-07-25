import os
import argparse
import numpy as np
from paz.pipelines.pose import SingleInstancePIX2POSE6D
from paz.models.segmentation import UNET_ConvNeXtBase
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from tensorflow.keras.utils import get_file
from paz.backend.image import show_image, load_image
from paz.processors import Processor
from paz.pipelines.detection import SSD300FAT
from paz.pipelines.pose import MultiInstancePIX2POSE6D


class SinglePowerDrillPIX2POSE6DUNet(SingleInstancePIX2POSE6D):
    """Predicts the pose6D of the YCB 035_power_drill object from an image.
        Optionally if a box2D message is given it translates the predicted
        points2D to new origin located at box2D top-left corner.

    # Arguments
        camera: PAZ Camera with intrinsic matrix.
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred points2D, points3D, pose6D and image.
    """
    def __init__(self, camera, epsilon=0.15, resize=False, draw=True):
        model = UNET_ConvNeXtBase(3, (224, 224, 3))
        name = 'model_weights.hdf5'
        directory_path = '/home/dfki.uni-bremen.de/pksharma/box/paz/examples/pix2pose/experiments/UNET-ConvNeXtBase_RUN_00_19-07-2023_16-02-50'
        weights_path = os.path.join(directory_path, name)
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path)
        object_sizes = np.array([1840, 1870, 520]) / 10000
        class_name = '035_power_drill'
        super(SinglePowerDrillPIX2POSE6DUNet, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


class MultiPowerDrillPIX2POSE6D(MultiInstancePIX2POSE6D):
    """Predicts poses6D of multiple instances the YCB 035_power_drill object
        from an image.

    # Arguments
        camera: PAZ Camera with intrinsic matrix.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, camera, offsets, epsilon=0.15, resize=False, draw=True):
        estimate_pose = SinglePowerDrillPIX2POSE6DUNet(
            camera, epsilon, resize, draw=False)
        super(MultiPowerDrillPIX2POSE6D, self).__init__(
            estimate_pose, offsets, camera, draw)


class PIX2POSEPowerDrill(Processor):
    """PIX2POSE inference pipeline with SSD300 trained on FAT and UNET-VGG16
        trained with domain randomization for the YCB object 035_power_drill.

    # Arguments
        score_thresh: Float between [0, 1] for object detector.
        nms_thresh: Float between [0, 1] indicating the non-maximum supression.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, camera, score_thresh=0.50, nms_thresh=0.45,
                 offsets=[0.25, 0.25], epsilon=0.15, resize=False, draw=True):
        self.detect = SSD300FAT(score_thresh, nms_thresh, draw=False)
        self.estimate_pose = MultiPowerDrillPIX2POSE6D(
            camera, offsets, epsilon, resize, draw)

    def call(self, image):
        return self.estimate_pose(image, self.detect(image)['boxes2D'])


parser = argparse.ArgumentParser(description='Object pose estimation')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')

args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
pipeline = PIX2POSEPowerDrill(camera, offsets=[0.15, 0.15], epsilon=0.015)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()


# URL = ('https://github.com/oarriaga/altamira-data/releases/download'
#        '/v0.9.1/image_with_YCB_objects.jpg')
# filename = os.path.basename(URL)
# fullpath = get_file(filename, URL, cache_subdir='paz/tests')
# image = load_image(fullpath)
# camera = Camera()
# camera.intrinsics_from_HFOV(55, image.shape)

# detect = PIX2POSEPowerDrill(camera, epsilon=0.015)
# inferences = detect(image)

# image = inferences['image']
# show_image(image)
