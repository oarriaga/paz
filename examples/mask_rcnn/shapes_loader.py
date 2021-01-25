import cv2
import numpy as np
from paz.backend.boxes import apply_non_max_suppression
from paz.abstract import Loader
from tensorflow.keras.utils import Progbar


class Shapes(Loader):
    """ Loader for shapes synthetic dataset.

    # Arguments
        num_samples: Int indicating number of samples to load
        size: (Height, Width) of input image to load
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name
    """
    def __init__(self, num_samples, size, split='train', class_names='all'):
        if class_names == 'all':
            class_names = ['square', 'circle', 'triangle']
        super(Shapes, self).__init__(None, split, class_names, 'Shapes')
        self.num_samples = num_samples
        self.size = size
        self.name_to_arg, self.arg_to_name = self._get_maps(self.class_names)

    def _get_maps(self, class_names):
        name_to_arg = dict(zip(class_names, range(len(class_names))))
        arg_to_name = dict(zip(range(len(class_names)), class_names))
        return name_to_arg, arg_to_name

    def load_data(self):
        progress_bar, data = Progbar(self.num_samples), []
        for sample_arg in range(self.num_samples):
            data.append(self.load_sample())
            progress_bar.update(sample_arg + 1)
        return data

    def load_sample(self):
        shapes = self.random_image()
        image = self.load_image(shapes)
        masks = self.load_masks(shapes)
        class_args = [self.name_to_arg[name[0]] for name in shapes]
        class_args = np.asarray(class_args).reshape(-1, 1)
        box_data = np.concatenate([self.get_boxes(masks), class_args], axis=1)
        labels = ['image', 'mask', 'box_data']
        sample = dict(zip(labels, [image, masks, box_data]))
        return sample

    def load_image(self, shapes):
        H, W = self.size
        background = np.array([np.random.randint(0, 255) for _ in range(3)])
        image = np.ones([H, W, 3], dtype=np.uint8)
        image = image * background.astype(np.uint8)
        for shape, color, dims in shapes:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def load_masks(self, shapes):
        H, W = self.size
        masks = np.zeros([H, W, len(shapes)], dtype=np.uint8)
        for idx, (shape, _, dims) in enumerate(shapes):
            masks[..., idx:idx+1] = self.draw_shape(
                masks[..., idx:idx+1].copy(), shape, dims, 1)
        occlusion = np.logical_not(masks[..., -1]).astype(np.uint8)
        for index in range(len(shapes)-2, -1, -1):
            masks[..., index] = masks[..., index] * occlusion
            occlusion = np.logical_and(occlusion,
                                       np.logical_not(masks[..., index]))
        return masks

    def get_boxes(self, masks):
        boxes = np.zeros([masks.shape[-1], 4], dtype=np.int32)
        for index in range(masks.shape[-1]):
            mask = masks[:, :, index]
            horizontal_indicies = np.where(np.any(mask, axis=0))[0]
            vertical_indicies = np.where(np.any(mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                X1, X2 = horizontal_indicies[[0, -1]]
                Y1, Y2 = vertical_indicies[[0, -1]]
                X2 += 1
                Y2 += 1
            else:
                X1, X2, Y1, Y2 = 0, 0, 0, 0
            boxes[index] = np.array([Y1, X1, Y2, X2])
        return boxes.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        center_x, center_y, size = dims
        if shape == 'square':
            start_point = (center_x - size, center_y - size)
            end_point = (center_x + size, center_y + size)
            cv2.rectangle(image, start_point, end_point, color, -1)
        elif shape == 'circle':
            cv2.circle(image, (center_x, center_y), size, color, -1)
        elif shape == 'triangle':
            angle = np.sin(np.radians(60))
            points = np.array([[(center_x, center_y - size),
                                (center_x - size / angle, center_y + size),
                                (center_x + size / angle, center_y + size),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width, buffer=20):
        shape = np.random.choice(self.class_names)
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        center_y = np.random.randint(buffer, height - buffer - 1)
        center_x = np.random.randint(buffer, width - buffer - 1)
        size = np.random.randint(buffer, height // 4)
        return shape, color, (center_x, center_y, size)

    def random_image(self):
        H, W = self.size
        shapes, boxes = [], []
        N = np.random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(H, W)
            shapes.append((shape, color, dims))
            X, Y, size = dims
            boxes.append([Y - size, X - size, Y + size, X + size])
        boxes = np.asarray(boxes)
        indices, _ = apply_non_max_suppression(boxes, np.arange(N), 0.3)
        shapes = [shape for _, shape in enumerate(shapes) if _ in indices]
        return shapes
