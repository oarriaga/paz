import numpy as np
from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from paz.backend.keypoints import build_cube_points3D, points3D_to_RGB
from paz.backend.image.draw import draw_circle


class PIX2POSE_ROCK(Processor):
    """Predicts pose6D from an RGB mask

    # Arguments
        estimate_pose: Function for estimating pose6D.
        offsets: Float between [0, 1] indicating ratio of increase of box2D.
        valid_class_names: List of strings indicating class names to be kept.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, estimate_pose, offsets, valid_class_names, draw=True):
        super(PIX2POSE_ROCK, self).__init__()
        self.estimate_pose = estimate_pose
        self.object_sizes = self.estimate_pose.object_sizes

        self.postprocess_boxes = SequentialProcessor([
            pr.FilterClassBoxes2D(valid_class_names),
            pr.SquareBoxes2D(),
            pr.OffsetBoxes2D(offsets)])

        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.unwrap = pr.UnwrapDictionary(['pose6D', 'points2D', 'points3D'])
        self.draw_boxes2D = pr.DrawBoxes2D(valid_class_names)
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw_pose6D = pr.DrawPose6D(self.cube_points3D,
                                         self.estimate_pose.camera.intrinsics)
        self.draw = draw
        self.wrap = pr.WrapOutput(['image', 'poses6D'])

    def call(self, image, boxes2D):
        boxes2D = self.postprocess_boxes(boxes2D)
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
                for point2D, color in zip(set_points2D, colors):
                    R, G, B = color
                    draw_circle(image, point2D.astype(np.int64),
                                (int(R), int(G), int(B)))
            for pose6D in poses6D:
                image = self.draw_pose6D(image, pose6D)
        return self.wrap(image, poses6D)


if __name__ == "__main__":
    from paz.backend.image import show_image, load_image
    from paz.backend.camera import Camera
    from paz.pipelines import RGBMaskToPowerDrillPose6D
    from paz.pipelines import SSD300FAT

    def approximate_intrinsics(image):
        image_size = image.shape[0:2]
        focal_length = image_size[1]
        image_center = (image_size[1] / 2.0, image_size[0] / 2.0)
        camera_intrinsics = np.array([[focal_length, 0, image_center[0]],
                                      [0, focal_length, image_center[1]],
                                      [0, 0, 1]])
        return camera_intrinsics

    image = load_image('images/test_image2.jpg')
    camera = Camera(device_id=0)
    camera.intrinsics = approximate_intrinsics(image)
    camera.distortion = np.zeros((4))

    estimate_pose = RGBMaskToPowerDrillPose6D(
        camera, 0.15, draw=False)
    pipeline = PIX2POSE_ROCK(
        estimate_pose, [0.5, 0.5], ['035_power_drill'], True)
    detect = SSD300FAT(0.5, 0.45, draw=False)

    boxes2D = detect(image)['boxes2D']
    inferences = pipeline(image, boxes2D)
    predicted_image = inferences['image']
    show_image(predicted_image)
