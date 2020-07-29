from paz.abstract import SequentialProcessor
from paz.abstract import Processor
from paz import processors as pr
from paz.pipelines import HaarCascadeFrontalFace
from paz.pipelines import FaceKeypointNet2D32
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


class EstimatePoseKeypoints(Processor):
    def __init__(self, detect, estimate_keypoints, camera,
                 offsets, model_points, class_to_dimensions, radius=3):
        super(EstimatePoseKeypoints, self).__init__()
        self.num_keypoints = estimate_keypoints.num_keypoints
        self.detect = detect
        self.estimate_keypoints = estimate_keypoints
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.solve_PNP = pr.SolvePNP(model_points, camera)
        self.draw_keypoints = pr.DrawKeypoints2D(self.num_keypoints, radius)
        self.draw_box3D = pr.DrawBoxes3D(camera, class_to_dimensions)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'keypoints', 'poses6D'])

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, keypoints2D = [], []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)['keypoints']
            keypoints = self.change_coordinates(keypoints, box2D)
            pose6D = self.solve_PNP(keypoints)
            image = self.draw_keypoints(image, keypoints)
            image = self.draw_box3D(image, pose6D)
            keypoints2D.append(keypoints)
            poses6D.append(pose6D)
        return self.wrap(image, boxes2D, keypoints2D, poses6D)


FACE_KEYPOINTS3D = np.array([
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

FACE_KEYPOINTS3D = FACE_KEYPOINTS3D - np.mean(FACE_KEYPOINTS3D, axis=0)


class HeadPoseKeypointNet2D32(EstimatePoseKeypoints):
    def __init__(self, camera, offsets=[0, 0], radius=3):
        detect = HaarCascadeFrontalFace(draw=False)
        estimate_keypoints = FaceKeypointNet2D32(draw=False)
        super(HeadPoseKeypointNet2D32, self).__init__(
            detect, estimate_keypoints, camera, offsets,
            FACE_KEYPOINTS3D, {None: [900.0, 600.0]}, radius=3)
