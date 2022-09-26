from paz.backend.image import show_image, write_image
from paz.pipelines import DetectSingleShot
from paz.models import SSD300
from efficientdet import EFFICIENTDETD0
from utils import get_class_name_efficientdet, raw_images
from paz.models import SSD300
from paz.datasets import get_class_names
from utils import efficientdet_preprocess
from efficientdet_postprocess import efficientdet_postprocess

if __name__ == "__main__":
    model = EFFICIENTDETD0(num_classes=21, base_weights='COCO', head_weights=None)
    model.load_weights("/home/manummk95/Desktop/efficientdet_working/temp/weight/weights.140-3.55.hdf5")    
    detections = DetectSingleShot(model, get_class_names('VOC'), 0.5, 0.45)(raw_images)
    show_image(detections['image'])
    write_image('latest_ops/new_op18.png', detections['image'])
    print('task completed')
