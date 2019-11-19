import os
import numpy as np
from paz.core import Processor
import paz.processors as pr
from paz.pipelines import SingleShotInference
from paz.pipelines import KeypointInference
from paz.models import SSD300
from paz.models import KeypointNet2D
from paz.datasets import get_class_names
from paz.core import VideoPlayer
from tensorflow.keras.utils import get_file


class PoseInference(Processor):
    def __init__(self, detector, keypoint_pipeline,
                 points3D, camera_intrinsics, distortions,
                 offset_scales=[.2, .2]):

        super(PoseInference, self).__init__()
        self.detector = detector
        self.detector.remove('DrawBoxes2D')
        self.detector.add(pr.FilterClassBoxes2D(['035_power_drill']))
        self.detector.add(pr.SquareBoxes2D())
        self.detector.add(pr.ClipBoxes2D())
        # self.detector.add(pr.CropBoxes2D(offset_scales, topic='image_crops'))
        self.offset_scales = offset_scales
        self.crop = pr.CropBoxes2D(self.offset_scales, topic='image_crops')

        self.keypoint_pipeline = keypoint_pipeline
        self.keypoint_pipeline.remove('DrawKeypoints2D')
        self.num_keypoints = self.keypoint_pipeline.num_keypoints

        self.points3D = points3D
        self.camera_intrinsics = camera_intrinsics
        self.distortions = distortions
        self.move_origin = pr.ChangeKeypointsCoordinateSystem()
        self.draw_boxes = pr.DrawBoxes2D(self.detector.class_names)
        self.draw_keypoints = pr.DrawKeypoints2D(self.num_keypoints, 3, False)
        self.solve_pnp = pr.SolvePNP(points3D, camera_intrinsics, distortions)

        self.show_image = pr.ShowImage()

    def call(self, kwargs):
        x = self.detector(kwargs)
        x = self.crop(x)
        # poses6D, keypoints = [], []
        for (box2D, image) in zip(x['boxes2D'], x['image_crops']):
            print(image.shape, image.max(), image.min())
            k = self.keypoint_pipeline({'box2D': box2D, 'image': image})
            # k = self.move_origin(k)
            # k = self.solve_pnp(k)
            k = self.draw_keypoints(k)
            return k
            # k = self.show_image(k)
            # poses6D.append(k['pose6D'])
            # keypoints.append(k['keypoints'])
        # x['poses6D'], x['keypoints'] = poses6D, keypoints
        # for keypoints in x['keypoints']:
        #     self.draw_keypoints({'image': x['image'], 'keypoints': keypoints})
        # x = self.draw_boxes(x)
        return x


score_thresh, nms_thresh, labels = .05, .45, get_class_names('FAT')

ssd300 = SSD300(len(labels), 'FAT', 'FAT')
detector = SingleShotInference(ssd300, labels, score_thresh, nms_thresh)
input_shape, num_keypoints = (128, 128, 3), 10
class_name = '035_power_drill'
keypointnet2D = KeypointNet2D(input_shape, num_keypoints)
model_name = '_'.join([keypointnet2D.name, str(num_keypoints), class_name])
filename = os.path.join(model_name, '_'.join([model_name, 'weights.hdf5']))
filename = get_file(filename, None, cache_subdir='paz/models')
keypointnet2D.load_weights(filename)
keypointer = KeypointInference(keypointnet2D, num_keypoints)

model_name = '_'.join(['keypointnet-shared', str(num_keypoints), class_name])
filename = os.path.join(model_name, 'keypoints_mean.txt')
filename = get_file(filename, None, cache_subdir='paz/models')
points3D = np.loadtxt(filename)[:, :3].astype(np.float64)

filename = os.path.join('logitech_c270', 'camera_intrinsics.txt')
filename = get_file(filename, None, cache_subdir='paz/cameras')
camera_intrinsics = np.loadtxt(filename)

filename = os.path.join('logitech_c270', 'distortions.txt')
filename = get_file(filename, None, cache_subdir='paz/cameras')
distortions = np.loadtxt(filename)

pose_pipeline = PoseInference(
    detector, keypointer, points3D, camera_intrinsics, distortions)

video_player = VideoPlayer((1280, 960), pose_pipeline, 2)
video_player.start()
