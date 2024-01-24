import argparse

from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt

parser = argparse.ArgumentParser(description='Visual Voice Activity Detection Live Demonstration')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = dt.DetectVVAD()
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
