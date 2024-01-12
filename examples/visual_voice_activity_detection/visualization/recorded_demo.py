import argparse

from paz.backend.camera import VideoPlayer, Camera
import paz.pipelines.detection as dt

parser = argparse.ArgumentParser(description='Visual Voice Activity Detection Recorded Demonstration')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

# model = pretrained_models.getFaceImageModel()

pipeline = dt.DetectVVAD()
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)

# player.record(name="./output/cedric.mp4", fourCC="MJPG", fps=25)

player.record_from_file(video_file_path="/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/paz/examples/visual_voice_activity_detection/visualization/output/cedric.mp4",
                        name="./output/cedric_labeled.avi", fps=25)
