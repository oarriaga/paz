import os

from matplotlib.pyplot import box
from paz import pipelines
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from paz.models import SSD300
from paz.pipelines import DetectSingleShot
from paz.applications import MinimalHandPoseEstimation
from paz.backend.camera import VideoPlayer, Camera
from model import SSD512Custom
from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
import cv2
import numpy as np
from paz.backend.image import lincolor


# weights_path = 'experiments/ADAM_WEIGHTS.hdf5'
weights_path = 'experiments/model_weights/tb_true/model_weights.hdf5'

class_names = ['background', 'hand']
model = SSD512Custom(2)
model.load_weights(weights_path)
score_thresh = 0.4
nms_thresh = 0.45


# detect = DetectSingleShot(model, class_names, score_thresh, nms_thresh)

class IsHandOpenPipeline(SequentialProcessor):
    def __init__(self, right_hand=False):
        super(IsHandOpenPipeline, self).__init__()
        self.add(MinimalHandPoseEstimation(right_hand))
        self.add(pr.UnpackDictionary(['keypoints2D', 'relative_angles']))
        self.add(pr.ControlMap(pr.IsHandOpen(), [1], [1]))
        self.add(pr.ControlMap(pr.BooleanToTextMessage('OPEN', 'CLOSE'), [1], [1]))
        self.add(pr.WrapOutput(['keypoints2D', 'status']))


class DetectMinimalHand(Processor):
    def __init__(self, offsets=[0, 0]):
        super(DetectMinimalHand, self).__init__()
        self.offsets = offsets
        self.croped_images = None
        self.hand_class_name = ['OPEN', 'CLOSE']
        self.colors = lincolor(len(self.hand_class_name))

        # detection
        self.detect = DetectSingleShot(model, class_names, score_thresh, nms_thresh)
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.hand_detector = IsHandOpenPipeline(right_hand=False)
        self.draw = pr.DrawBoxes2D(self.hand_class_name, self.colors,
                                   weighted=True, with_score=False)
        self.draw_skeleton = pr.DrawHandSkeleton()
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image.copy())['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            cropped_image = np.array(cropped_image)
            cropped_image = np.squeeze(cropped_image)
            inference = self.hand_detector(cropped_image)
            box2D.class_name = inference['status']
            keypoints = inference['keypoints2D']
            keypoints = np.array(keypoints) + np.array(box2D.coordinates[:2])
        image = self.draw_skeleton(image, keypoints)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)



pipeline = DetectMinimalHand()
camera = Camera(device_id=0)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
# player.record_from_file('./minimal_hand_video1.mp4')
# player.record()