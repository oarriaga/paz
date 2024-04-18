import os
import numpy as np
from paz.abstract import Loader


class TXTLoader(Loader):
    """Preprocess the .txt annotations data.

    # Arguments
        path: path of the annotation.txt

    # Return
        data: dictionary with keys corresponding to the image paths
              and values numpy arrays of shape "[num_objects, 4+1]"
    """

    def __init__(self, path, class_names, split='train'):
        super(TXTLoader, self).__init__(path, split, class_names, 'TXTLoader')
        self.class_to_arg = self.build_class_to_arg(self.class_names)

    def build_class_to_arg(self, class_names):
        args = list(range(len(class_names)))
        return dict(zip(class_names, args))

    def load_data(self):
        images_path = os.path.dirname(self.path)
        data = []

        file = open(self.path, 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            """
             get the image name (write in the data),
             image size (to normalize the bounding box, not normalize first),
             class name, bounding box (write in data)
            """
            image_name, H, W, class_name, x_min, y_min, x_max, y_max = line.split(',')
            image_path = os.path.join(images_path, image_name)
            box_data = []
            box_data.append([float(x_min) / int(W), float(y_min) / int(H),
                             float(x_max) / int(W), float(y_max) / int(H),
                             self.class_to_arg.get(class_name)])
            box_data = np.asarray(box_data)
            data.append({'image': image_path, 'boxes': box_data})
        return data


if __name__ == "__main__":
    path = 'training_images/images/annotation.txt'
    class_names = ['background', 'coyote']
    data_manager = TXTLoader(path, class_names)
    dataset = data_manager.load_data()
    print(dataset)
