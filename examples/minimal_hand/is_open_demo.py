import argparse
from paz.applications import MinimalHandPoseEstimation
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.abstract import SequentialProcessor
from paz import processors as pr

parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()


pipeline = SequentialProcessor()
pipeline.add(MinimalHandPoseEstimation(right_hand=False))
pipeline.add(pr.UnpackDictionary(['image', 'relative_angles']))
pipeline.add(pr.ControlMap(pr.IsHandOpen(), [1], [1]))
pipeline.add(pr.ControlMap(pr.BooleanToTextMessage('OPEN', 'CLOSE'), [1], [1]))
pipeline.add(pr.ControlMap(pr.DrawText(), [0, 1], [1]))
pipeline.add(pr.WrapOutput(['image', 'status']))

camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
