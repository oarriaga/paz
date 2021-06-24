import os
import glob
from paz.pipelines import DetectSingleShot
from paz.models.detection import SSD300
from paz.backend.image import load_image, show_image, write_image


class SSD300SolarPanel(DetectSingleShot):
    def __init__(self, weights_path, score_thresh=0.50,
                 nms_thresh=0.45, draw=True):
        class_names = ['background', 'solar_panel']
        model = SSD300(len(class_names), None, None)
        model.load_weights(weights_path)
        super(SSD300SolarPanel, self).__init__(
            model, class_names, score_thresh, nms_thresh, draw=draw)


# weights_path = 'trained_models/SSD300/weights.172-3.15.hdf5'
weights_path = 'trained_models/SSD300/weights.141-2.66.hdf5'
detect = SSD300SolarPanel(weights_path)
image_paths = glob.glob('datasets/test_solar_panel/*.jpg')
for image_arg, image_path in enumerate(image_paths):
    image = load_image(image_path)
    results = detect(image)
    # show_image(results['image'])
    write_image('results/image_%s.png' % image_arg, results['image'])
