from paz.core import VideoPlayer
from paz.pipelines import SingleShotInference
from paz.models import SSD300
from paz.datasets import get_class_names
# import pickle


score_thresh, nms_thresh, labels = .05, .45, get_class_names('VOC')
# model = SSD300(len(labels), None, None)
# weights_path = '/home/octavio/altamira_weights.pkl'
# weights = pickle.load(open(weights_path, 'rb'))
# model.set_weights(weights)
# model.summary()
model = SSD300(len(labels))
# model.load_weights('weights.77-4.48.hdf5')
detector = SingleShotInference(model, labels, score_thresh, nms_thresh)
video_player = VideoPlayer((1280, 960), detector, 0)
video_player.start()

"""
from paz.processors import ShowImage
from paz.processors import Resize
from paz.core import ops

detector.add(Resize((240, 480)))
detector.add(ShowImage())

image = ops.load_image('test_image.jpg')
detector(image=image)
"""
