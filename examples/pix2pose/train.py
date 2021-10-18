import os
import glob
from paz.abstract import GeneratingSequence
from paz.backend.image import show_image

from scenes import PixelMaskRenderer
from pipelines import DomainRandomization
from loss import WeightedForeground

image_shape = [128, 128, 3]
root_path = os.path.expanduser('~')
background_wildcard = '.keras/paz/datasets/voc-backgrounds/*.png'
background_wildcard = os.path.join(root_path, background_wildcard)
image_paths = glob.glob(background_wildcard)
path_OBJ = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
path_OBJ = os.path.join(root_path, path_OBJ)
num_occlusions = 1
viewport_size = image_shape[:2]
y_fov = 3.14159 / 4.0
distance = [0.3, 0.5]
light = [1.0, 30]
top_only = False
roll = 3.14159
shift = 0.05
num_steps = 1000
batch_size = 32
beta = 3.0


renderer = PixelMaskRenderer(path_OBJ, viewport_size, y_fov, distance,
                             light, top_only, roll, shift)

processor = DomainRandomization(renderer, image_shape,
                                image_paths, num_occlusions)

sequence = GeneratingSequence(processor, batch_size, num_steps)

weighted_foreground = WeightedForeground(beta)

# batch = sequence.__getitem__(0)
# for _ in range(100):
# image, alpha, RGB_mask = renderer.render()
# show_image(image)
# show_image(RGB_mask)


"""
for _ in range(100):
    sample = processor()
    inputs, labels = sample['inputs'], sample['labels']
    show_image((inputs['input_image'] * 255).astype('uint8'))
    show_image((labels['label_image'] * 255).astype('uint8'))
"""
