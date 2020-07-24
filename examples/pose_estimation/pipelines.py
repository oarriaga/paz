from paz.abstract import SequentialProcessor
from paz.abstract import Processor
from paz import processors as pr
import numpy as np


class AugmentKeypoints(SequentialProcessor):
    def __init__(self, phase, rotation_range=30,
                 delta_scales=[0.2, 0.2], num_keypoints=15):
        super(AugmentKeypoints, self).__init__()

        self.add(pr.UnpackDictionary(['image', 'keypoints']))
        if phase == 'train':
            self.add(pr.ControlMap(pr.RandomBrightness()))
            self.add(pr.ControlMap(pr.RandomContrast()))
            self.add(pr.RandomKeypointRotation(rotation_range))
            self.add(pr.RandomKeypointTranslation(delta_scales))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(pr.ExpandDims(-1), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeKeypoints((96, 96)), [1], [1]))
        self.add(pr.SequenceWrapper({0: {'image': [96, 96, 1]}},
                                    {1: {'keypoints': [num_keypoints, 2]}}))


class HeadPose6DEstimation(Processor):
    def __init__(self, detector, keypointer, model_points, camera, radius=3):
        super(HeadPose6DEstimation, self).__init__()
        # face detector
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.detect = pr.Predict(detector, RGB2GRAY, pr.ToBoxes2D(['face']))

        # creating pre-processing pipeline for keypoint estimator
        preprocess = SequentialProcessor()
        preprocess.add(pr.ResizeImage(keypointer.input_shape[1:3]))
        preprocess.add(pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.NormalizeImage())
        preprocess.add(pr.ExpandDims([0, 3]))

        # prediction
        self.estimate_keypoints = pr.Predict(
            keypointer, preprocess, pr.Squeeze(0))

        # used for drawing up keypoints in original image
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.denormalize_keypoints = pr.DenormalizeKeypoints()
        self.crop_boxes2D = pr.CropBoxes2D()
        self.num_keypoints = keypointer.output_shape[1]
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, radius, False)
        self.draw_box3D = pr.DrawBoxes3D(camera, model_points['dimensions'])
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

        self.solve_PNP = pr.SolvePNP(model_points['keypoints3D'], camera)

    def call(self, image):
        boxes2D = self.detect(image)
        poses6D = []
        cropped_images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)
            keypoints = self.denormalize_keypoints(keypoints, cropped_image)
            keypoints = self.change_coordinates(keypoints, box2D)
            pose6D = self.solve_PNP(keypoints)
            image = self.draw_box3D(image, pose6D)
            image = self.draw(image, keypoints)
            poses6D.append(pose6D)
        return self.wrap(image, boxes2D, poses6D)


keypoints3D = np.array([
    [-220, 678, 1138],  # left--center-eye
    [+220, 678, 1138],  # right-center-eye
    [-131, 676, 1107],  # left--eye close to nose
    [-294, 610, 1123],  # left--eye close to ear
    [+131, 676, 1107],  # right-eye close to nose
    [+294, 610, 1123],  # right-eye close to ear
    [-106, 758, 1224],  # left--eyebrow close to nose
    [-375, 585, 1208],  # left--eyebrow close to ear
    [+106, 758, 1224],  # right-eyebrow close to nose
    [+375, 585, 1208],  # right-eyebrow close to ear
    [0.0, 909, 919],  # nose
    [-183, 691, 683],  # lefty-lip
    [+183, 691, 683],  # right-lip
    [0.0, 826, 754],  # up---lip
    [0.0, 815, 645],  # down-lip
])


keypoints3D = keypoints3D - np.mean(keypoints3D, axis=0)
model_data = {'keypoints3D': keypoints3D, 'dimensions': {None: [900.0, 600.0]}}
