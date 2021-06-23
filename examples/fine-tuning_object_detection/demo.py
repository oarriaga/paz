import os
import glob
from paz.pipelines import DetectSingleShot
from paz.models.detection import SSD300
from paz.backend.image import load_image, show_image


class SSD300SolarPanel(DetectSingleShot):
    def __init__(self, weights_path, score_thresh=0.30,
                 nms_thresh=0.45, draw=True):
        class_names = ['background', 'solar_panel']
        model = SSD300(len(class_names), None, None)
        model.load_weights(weights_path)
        DetectSingleShot(model, class_names, 0.6, 0.45)
        super(SSD300SolarPanel, self).__init__(
            model, class_names, score_thresh, nms_thresh, draw=draw)


weights_path = 'trained_models/SSD300/weights.172-3.15.hdf5'
detect = SSD300SolarPanel(weights_path)
image_paths = glob.glob('datasets/test_solar_panel/*.jpg')
for image_path in image_paths:
    image = load_image(image_path)
    results = detect(image)
    show_image(results['image'])
