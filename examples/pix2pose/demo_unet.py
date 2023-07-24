import os
import argparse
import numpy as np
from paz.pipelines.pose import SingleInstancePIX2POSE6D
from paz.models.segmentation import UNET_ConvNeXtBase
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera



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
        model = UNET_ConvNeXtBase(1, (128, 128, 3))
        name = 'UNET-VGG16_POWERDRILL_weights.hdf5'
        directory_path = '/home/dfki.uni-bremen.de/pksharma/box/paz/examples/pix2pose/experiments/UNET-ConvNeXtBase_RUN_00_19-07-2023_16-02-50'
        weights_path = os.path.join(directory_path, name)
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path)
        object_sizes = np.array([1840, 1870, 520]) / 10000
        class_name = '035_power_drill'
        super(SinglePowerDrillPIX2POSE6DUNet, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


parser = argparse.ArgumentParser(description='Object pose estimation')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera(args.camera_id)
pipeline = SinglePowerDrillPIX2POSE6DUNet(camera, epsilon=0.015)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
