import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from bounding_boxes import draw_bounding_box

image = mpimg.imread('/media/fabian/Data/Masterarbeit/data/VOCdevkit/VOC2012/JPEGImages/2008_000002.jpg')

image = draw_bounding_box(image, np.array([int(0.066*500), int(375*0.02666667), int(500*0.894), int(375*0.77866667)]))

plt.imshow(image)
plt.show()