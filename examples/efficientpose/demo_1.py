from paz.backend.image import load_image, show_image
from pose import EFFICIENTPOSEALINEMODDRILLER

IMAGE_PATH = ('/home/manummk95/Desktop/paz/paz/examples/efficientpose/'
              'dataset_less/Linemod_preprocessed/data/08/rgb/0002.png')


detect = EFFICIENTPOSEALINEMODDRILLER(score_thresh=0.85, nms_thresh=0.010,
                                      show_boxes2D=True, show_poses6D=True)
detect.model.load_weights('weights.3498-22.03.hdf5')
image = load_image(IMAGE_PATH)
inferences = detect(image)
show_image(inferences['image'])
