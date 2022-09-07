import argparse
from paz.abstract import SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
from paz.applications import SSD512MinimalHandPose
from paz import processors as pr


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
focal_length = camera.intrinsics[0, 0]


class EstimateTranslation3D(pr.Processor):
    def __init__(self, camera, real_width=10):
        super(EstimateTranslation3D, self).__init__()
        self.camera = camera
        self.focal_length = self.camera.intrinsics[0, 0]
        self.real_width = real_width
        self.u_camera_center = self.camera.intrinsics[0, 2]
        self.v_camera_center = self.camera.intrinsics[1, 2]

    def call(self, boxes2D):
        hands_center = []
        for box in boxes2D:
            u_box_center, v_box_center = box.center
            z_center = (self.real_width * focal_length) / box.width
            u = u_box_center - self.u_camera_center
            v = v_box_center - self.v_camera_center
            x_center = (z_center * u) / self.focal_length
            y_center = (z_center * v) / self.focal_length
            hands_center.append([x_center, y_center, z_center])
        print(hands_center)
        return hands_center


pipeline = SequentialProcessor()
pipeline.add(SSD512MinimalHandPose())
pipeline.add(pr.UnpackDictionary(['image', 'boxes2D', 'keypoints2D']))
pipeline.add(pr.ControlMap(EstimateTranslation3D(camera), [1], [1]))
pipeline.add(pr.WrapOutput(['image', 'position']))


player = VideoPlayer((640, 480), pipeline, camera)
player.run()
