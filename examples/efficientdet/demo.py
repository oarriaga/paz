from tensorflow.keras.utils import get_file
from paz.backend.image import show_image, write_image
from paz.datasets import get_class_names
from paz.pipelines.detection import DetectSingleShot
from paz.processors.image import LoadImage
from detection import DrawBoxes2D
from efficientdet import EFFICIENTDETD0

IMAGE_PATH = ('/home/manummk95/Desktop/efficientdet_BKP/paz/'
              'examples/efficientdet/000132.jpg')
WEIGHT_PATH = (
    'https://github.com/oarriaga/altamira-data/releases/download/v0.16/')
WEIGHT_FILE = 'efficientdet-d0-VOC-VOC_weights.hdf5'

if __name__ == "__main__":
    raw_image = LoadImage()(IMAGE_PATH)
    model = EFFICIENTDETD0(num_classes=21, base_weights='COCO',
                           head_weights=None)
    weights_path = get_file(WEIGHT_FILE, WEIGHT_PATH + WEIGHT_FILE,
                            cache_subdir='paz/models')
    model.load_weights(weights_path)

    detect = DetectSingleShot(model, get_class_names('VOC'), 0.5, 0.45)
    detect.draw_boxes2D = DrawBoxes2D(get_class_names('VOC'))
    detections = detect(raw_image)

    show_image(detections['image'])
    write_image('predictions.png', detections['image'])
    print('task completed')
