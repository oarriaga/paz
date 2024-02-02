from paz.backend.camera import VideoPlayer
import paz.pipelines.detection as dt

pipeline = dt.DetectVVAD()
player = VideoPlayer((640, 480), pipeline, None)

player.record_from_file(video_file_path="./demo_video.mp4",
                        name="./demo_video_labeled.avi", fps=25)
