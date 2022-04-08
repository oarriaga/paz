import numpy as np
import processors as pe
from paz import processors as pr
from paz.backend.image import flip_left_right
from models.detnet import DetNet


class HandPoseEstimation(pr.Processor):
    def __init__(self, hand_estimator, size=128, draw=True):
        super(HandPoseEstimation).__init__()
        self.size = size
        self.draw = draw
        self.resize_image = pr.ResizeImage((size, size))
        self.get_scaling_factor = pe.GetScalingFactor(scaling_factor=4)
        self.expand_dims = pr.ExpandDims(axis=0)
        self.hand_estimator = hand_estimator
        self.draw_skeleton = pe.DrawHandSkeleton()
        self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D'])

    def call(self, input_image, flip_input_image=False):
        image = self.resize_image(input_image)
        scale = self.get_scaling_factor(input_image, self.size)

        if flip_input_image:
            image = flip_left_right(image)

        image = self.expand_dims(image).astype(np.float32)
        keypoints3D, keypoints2D = self.hand_estimator.predict(image)[:2]

        keypoints2D = flip_left_right(keypoints2D)
        keypoints2D = np.array(keypoints2D*scale, dtype=np.uint)
        if self.draw:
            image = self.draw_skeleton(input_image, keypoints2D)
        return self.wrap(image, keypoints3D, keypoints2D)


class MANOHandPoseEstimation(HandPoseEstimation):
    def __init__(self):
        detect_hand = DetNet()
        super(MANOHandPoseEstimation, self).__init__(detect_hand)
