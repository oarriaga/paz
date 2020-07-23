from paz.backend.keypoints import denormalize_keypoints, solve_PNP, UPNP
from paz.abstract import SequentialProcessor, Pose6D
from paz.abstract import Processor
from paz import processors as pr
from paz.backend.image import draw_circle
from paz.backend.image.draw import GREEN
import numpy as np
import cv2

def draw_circles(image, points, color=GREEN, radius=3):
    for point in points:
        draw_circle(image, point, color, radius)
    return image


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


class SolvePNP(Processor):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.

    # Arguments
        model_points: Numpy array of shape ''(num_points, 3)''.
            Model 3D points known in advance.
        camera: Instance of ''paz.backend.Camera'' containing as properties
            the ''camera_intrinsics'' a Numpy array of shape ''(3, 3)''
            usually calculated from the openCV ''calibrateCamera'' function,
            and the ''distortion'' a Numpy array of shape ''(5)'' in which the
            elements are usually obtained from the openCV
            ''calibrateCamera'' function.

    # Returns
        Instance from ''Pose6D'' message.
    """
    def __init__(self, points3D, camera, class_name=None):
        super(SolvePNP, self).__init__()
        self.points3D = points3D
        self.camera = camera
        self.class_name = class_name
        self.num_keypoints = len(points3D)

    def call(self, keypoints):
        keypoints = keypoints[:, :2]
        keypoints = keypoints.astype(np.float64)
        keypoints = keypoints.reshape((self.num_keypoints, 1, 2))
        # solver = UPNP
        solver = cv2.SOLVEPNP_ITERATIVE
        (success, rotation, translation) = solve_PNP(
            self.points3D, keypoints, self.camera, solver)

        pose6D = Pose6D.from_rotation_vector(
            rotation, translation, self.class_name)
        return pose6D


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
        self.draw_boxes2D = pr.DrawBoxes2D(['face'], colors=[[0, 255, 0]])
        self.draw_box3D = pr.DrawBoxes3D(camera, model_points['dimensions'])
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

        self.solve_PNP = SolvePNP(model_points['keypoints3D'], camera)

    def call(self, image):
        boxes2D = self.detect(image)
        poses6D = []
        cropped_images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)
            keypoints = self.denormalize_keypoints(keypoints, cropped_image)
            keypoints = self.change_coordinates(keypoints, box2D)
            # keypoints = keypoints[13:15, :]
            keypoints = keypoints[[3, 5, 10, 11, 12, 14], :]
            pose6D = self.solve_PNP(keypoints)
            image = self.draw_box3D(image, pose6D)
            image = self.draw(image, keypoints)
            poses6D.append(pose6D)
        # image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D, poses6D)


if __name__ == '__main__':
    from paz.abstract import ProcessingSequence
    from paz.backend.image import show_image

    from facial_keypoints import FacialKeypoints

    data_manager = FacialKeypoints('dataset/', 'train')
    dataset = data_manager.load_data()
    augment_keypoints = AugmentKeypoints('train')
    for arg in range(1, 100):
        sample = dataset[arg]
        predictions = augment_keypoints(sample)
        original_image = predictions['inputs']['image'][:, :, 0]
        original_image = original_image * 255.0
        kp = predictions['labels']['keypoints']
        kp = denormalize_keypoints(kp, 96, 96)
        original_image = draw_circles(
            original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
    sequence = ProcessingSequence(augment_keypoints, 32, dataset, True)
    batch = sequence.__getitem__(0)
