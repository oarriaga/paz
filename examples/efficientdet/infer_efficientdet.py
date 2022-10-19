from paz.backend.image import show_image, write_image
from paz.datasets import get_class_names

from detection import DetectSingleShot
from efficientdet import EFFICIENTDETD0
from utils import raw_images


if __name__ == "__main__":
    model = EFFICIENTDETD0(num_classes=21, base_weights='COCO',
                           head_weights=None)
    model.load_weights("/home/manummk95/Desktop/efficientdet_working/temp/"
                       "weight/weights.209-3.73.hdf5")
    detections = DetectSingleShot(model, get_class_names('VOC'),
                                  0.5, 0.45)(raw_images)
    show_image(detections['image'])
    write_image('predictions.png', detections['image'])
    print('task completed')
