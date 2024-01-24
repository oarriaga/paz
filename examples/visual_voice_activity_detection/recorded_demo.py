from paz.backend.camera import VideoPlayer
import paz.pipelines.detection as dt

pipeline = dt.DetectVVAD()
player = VideoPlayer((640, 480), pipeline, None)

player.record_from_file(video_file_path="visualization/output/cedric.mp4",
                        name="./output/cedric_labeled.avi", fps=25)
