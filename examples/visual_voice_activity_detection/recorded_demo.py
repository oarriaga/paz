import argparse

from paz.backend.camera import VideoPlayer
import paz.pipelines.detection as dt

parser = argparse.ArgumentParser(description='Visual Voice Activity Detection Recorded Demonstration')
parser.add_argument('-i', '--input_path', type=str, default="./demo_video.mp4",
                    help='Path to the video file to be used as input for the VVAD Pipeline.')
parser.add_argument('-o', '--output_path', type=str, default="./demo_video_labeled.avi",
                    help='Path to the video file to be used as output for the VVAD Pipeline.')
args = parser.parse_args()

pipeline = dt.DetectVVAD()
player = VideoPlayer((640, 480), pipeline, None)

player.record_from_file(video_file_path=args.input_path,
                        name=args.output_path, fps=25)
