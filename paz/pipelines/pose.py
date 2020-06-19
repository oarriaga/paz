from ..abstract import Processor
from .. import processors as pr

from .keypoints import PredictKeypoints


class PredictPoseFromKeypoints(Processor):
    def __init__(self, model, points3D, camera, class_to_dimensions, radius=5):

        super(PredictPoseFromKeypoints, self).__init__()
        self.num_keypoints = model.output_shape[1]
        self.radius = radius
        self.predict_keypoints = PredictKeypoints(model)
        self.predict_keypoints.add(pr.SolvePNP(points3D, camera))
        self.draw = pr.DrawBoxes3D(camera, class_to_dimensions)
        self.wrap = pr.WrapOutput(['image', 'keypoints'])

    def call(self, image):
        keypoints = self.predict_keypoints(image)
        image = self.draw(image, keypoints)
        return self.wrap(image, keypoints)
