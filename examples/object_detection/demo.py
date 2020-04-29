from paz.backend.camera import VideoPlayer, Camera
from paz.models import SSD300
from paz.datasets import get_class_names
from paz.pipelines import SingleShotInference


score_thresh, nms_thresh, labels = .6, .45, get_class_names('VOC')
model = SSD300(len(labels))
detector = SingleShotInference(model, labels, score_thresh, nms_thresh)
camera = Camera(device_id=0)
video_player = VideoPlayer((1280, 960), detector, camera)
video_player.run()
