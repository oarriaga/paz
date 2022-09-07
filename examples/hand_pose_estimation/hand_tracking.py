import argparse
from paz.abstract import SequentialProcessor
from paz.backend.camera import VideoPlayer, Camera
from paz.applications import SSD512MinimalHandPose
from paz import processors as pr


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=4,
                    help='Camera device ID')
parser.add_argument('-HFOV', '--horizontal_field_of_view', type=float,
                    default=75, help='Horizontal field of view in degrees')
args = parser.parse_args()

camera = Camera(args.camera_id)
camera.intrinsics_from_HFOV(args.horizontal_field_of_view)
focal_length = camera.intrinsics[0, 0]


class Translation3DFromBoxWidth(pr.Processor):
    def __init__(self, camera, real_width=30):
        super(Translation3DFromBoxWidth, self).__init__()
        self.camera = camera
        self.real_width = real_width
        self.focal_length = self.camera.intrinsics[0, 0]
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
        return hands_center


class PrintTopics(pr.Processor):
    def __init__(self, topics):
        super(PrintTopics, self).__init__()
        self.topics = topics

    def call(self, dictionary):
        [print(dictionary[topic]) for topic in self.topics]
        return dictionary


class HandStateEstimation(SequentialProcessor):
    def __init__(self, camera):
        super(HandStateEstimation, self).__init__()
        intro_topics = ['image', 'boxes2D', 'keypoints2D', 'keypoints3D']
        self.add(SSD512MinimalHandPose())
        self.add(pr.UnpackDictionary(intro_topics))
        self.add(pr.ControlMap(Translation3DFromBoxWidth(camera), [1], [4], {1: 1}))
        outro_topics = intro_topics + ['translation3D']
        self.add(pr.WrapOutput(outro_topics))


pipeline = HandStateEstimation(camera)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
