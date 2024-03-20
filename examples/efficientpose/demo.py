from paz.backend.image import load_image, show_image
from pose import EfficientPosePhi0LinemodDriller

IMAGE_PATH = ('/home/manummk95/Desktop/paz/paz/examples/efficientpose/'
              'Linemod_preprocessed/data/08/rgb/0002.png')


detect = EfficientPosePhi0LinemodDriller(score_thresh=0.60, nms_thresh=0.45,
                                         show_boxes2D=True, show_poses6D=True)
detect.model.load_weights('weights.6760-1.39.hdf5')
image = load_image(IMAGE_PATH)
inferences = detect(image)
show_image(inferences['image'])
