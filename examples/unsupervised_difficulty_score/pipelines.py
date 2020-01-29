from paz.core import SequentialProcessor
from paz import processors as pr


class ImageAugmentation(SequentialProcessor):
    def __init__(self, size, num_classes, split='train'):
        super(ImageAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')

        self.size = size
        self.num_classes = num_classes
        self.split = split

        self.add(pr.CastImageToFloat())
        self.add(pr.ResizeImage((self.size, self.size)))
        self.add(pr.ExpandDims(axis=-1, topic='image'))
        self.add(pr.NormalizeImage())
        self.add(pr.OutputSelector(['image'], ['label']))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 1)]

    @property
    def label_shapes(self):
        return [(self.num_classes, )]
