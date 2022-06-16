import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from paz.models import SSD300
from paz.pipelines import DetectSingleShot
from paz.backend.camera import VideoPlayer, Camera
from model import SSD512Custom

# weights_path = 'experiments/ADAM_WEIGHTS.hdf5'
weights_path = 'experiments/SSD512Custom_RUN_00_13-06-2022_08-10-10/model_weights.hdf5'

class_names = ['background', 'hand']
model = SSD512Custom(2)
# model = SSD300(len(class_names), None, None)
model.load_weights(weights_path)
score_thresh = 0.4

nms_thresh = 0.45
detect = DetectSingleShot(model, class_names, score_thresh, nms_thresh)

camera = Camera(device_id=4)
player = VideoPlayer((1280, 960), detect, camera)
player.run()
