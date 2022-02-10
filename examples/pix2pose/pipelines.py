from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor, Pose6D
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.pipelines import Pix2Points
from paz.backend.groups.quaternion import rotation_vector_to_quaternion
from paz.backend.keypoints import build_cube_points3D
from paz.backend.keypoints import denormalize_keypoints2D
from paz.backend.image.draw import draw_points2D
from paz.backend.keypoints import points3D_to_RGB


class DomainRandomization(SequentialProcessor):
    """Performs domain randomization on a rendered image
    """
    def __init__(self, renderer, image_shape, image_paths, inputs_to_shape,
                 labels_to_shape, num_occlusions=1):
        super(DomainRandomization, self).__init__()
        H, W = image_shape[:2]
        self.add(pr.Render(renderer))
        self.add(pr.ControlMap(RandomizeRender(image_paths), [0, 1], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [1], [1]))
        self.add(pr.SequenceWrapper({0: inputs_to_shape},
                                    {1: labels_to_shape}))


class Pix2Pose(pr.Processor):
    def __init__(self, model, object_sizes, camera, epsilon=0.15,
                 resize=False, class_name=None, draw=True):

        self.model = model
        self.resize = resize
        self.object_sizes = object_sizes
        self.camera = camera
        self.epsilon = epsilon
        self.class_name = str(class_name) if class_name is None else class_name
        self.draw = draw

        self.predict_points = Pix2Points(
            self.model, self.object_sizes, self.epsilon, self.resize)
        self.predict_pose = pr.SolveChangingObjectPnPRANSAC(camera.intrinsics)
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw_pose6D = pr.DrawPose6D(self.cube_points3D,
                                         self.camera.intrinsics)

    def call(self, image, box2D=None):
        results = self.predict_points(image)
        points2D, points3D = results['points2D'], results['points3D']
        H, W = image.shape[:2]
        points2D = denormalize_keypoints2D(points2D, H, W)

        if box2D is not None:
            points2D = self.change_coordinates(points2D, box2D)
            self.class_name = box2D.class_name

        if len(points3D) > self.predict_pose.MIN_REQUIRED_POINTS:
            success, R, translation = self.predict_pose(points3D, points2D)
            if success:
                quaternion = rotation_vector_to_quaternion(R)
                pose6D = Pose6D(quaternion, translation, self.class_name)
            else:
                pose6D = None
        else:
            pose6D = None

        # box2D check required since change_coordinates goes outside (crop) img
        if (self.draw and (box2D is None) and (pose6D is not None)):
            colors = points3D_to_RGB(points3D, self.object_sizes)
            image = draw_points2D(image, points2D, colors)
            image = self.draw_pose6D(image, pose6D)
            results['image'] = image
        results['points2D'], results['pose6D'] = points2D, pose6D
        return results


class EstimatePoseMasks(Processor):
    def __init__(self, detect, estimate_pose, offsets, draw=True,
                 valid_class_names=['035_power_drill']):
        """Pose estimation pipeline using keypoints.
        """
        super(EstimatePoseMasks, self).__init__()
        self.detect = detect
        self.estimate_pose = estimate_pose
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(valid_class_names),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.unwrap = pr.UnwrapDictionary(['pose6D', 'points2D', 'points3D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.object_sizes = self.estimate_pose.object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw_pose6D = pr.DrawPose6D(
            self.cube_points3D, self.estimate_pose.camera.intrinsics)
        self.draw = draw

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points2D, points3D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            results = self.estimate_pose(crop, box2D)
            pose6D, set_points2D, set_points3D = self.unwrap(results)
            points2D.append(set_points2D), points3D.append(set_points3D)
            poses6D.append(pose6D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            for set_points2D, set_points3D in zip(points2D, points3D):
                colors = points3D_to_RGB(set_points3D, self.object_sizes)
                image = draw_points2D(image, set_points2D, colors)
            for pose6D in poses6D:
                image = self.draw_pose6D(image, pose6D)
        return self.wrap(image, boxes2D, poses6D)
