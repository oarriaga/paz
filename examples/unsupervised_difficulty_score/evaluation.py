import h5py
import numpy as np
import matplotlib.pyplot as plt

from datasets import MNIST


def make_mosaic(images, titles=None, rows=1):
    assert((titles is None) or (len(images) == len(titles)))
    num_images = len(images)
    figure = plt.figure()
    for image_arg, (image, title) in enumerate(zip(images, titles)):
        cols = np.ceil(num_images / float(rows))
        axis = figure.add_subplot(rows, cols, image_arg + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        if titles is not None:
            axis.set_title(title)
    figure.set_size_inches(np.array(figure.get_size_inches()) * num_images)
    return figure


top_k, evaluation_arg = 9, 0
data = MNIST('test').load()

evaluations = h5py.File('evaluations/MNIST.hdf5', 'r')
evaluations = np.asarray(evaluations['evaluations'])
evaluations = evaluations[..., :len(data), evaluation_arg]

action_scores = np.sum(evaluations, axis=0)
sorted_scores = np.argsort(action_scores)

print('Displaying easiest samples')
images, titles = [], []
for arg in sorted_scores[:top_k]:
    print(action_scores[arg], arg)
    images.append(data[arg]['image'])
    action_score = round(action_scores[arg], 2)
    titles.append(str(action_score))

figure = make_mosaic(images, titles, rows=3)
figure.suptitle('Easiest examples', fontsize=16)
plt.show()

print('Displaying easiest samples')
images, titles = [], []
for arg in sorted_scores[-top_k:]:
    print(action_scores[arg], arg)
    images.append(data[arg]['image'])
    action_score = round(action_scores[arg], 2)
    titles.append(str(action_score))


figure = make_mosaic(images, titles, rows=3)
figure.suptitle('Most difficult samples', fontsize=16)
plt.show()
