import argparse

from paz.backend.camera import VideoPlayer
import paz.pipelines.detection as dt

parser = argparse.ArgumentParser(description='Visual Voice Activity Detection Recorded Demonstration')
parser.add_argument('-p', '--path', type=str, default="./demo_video.mp4",
                    help='Path to the video file to be used as input for the VVAD Pipeline.')
args = parser.parse_args()

pipeline = dt.DetectVVAD()
player = VideoPlayer((640, 480), pipeline, None)

player.record_from_file(video_file_path="args.path",
                        name="./demo_video_labeled.avi", fps=25)
