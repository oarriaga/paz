from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import EncoderPredictor, DecoderPredictor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr


class DomainRandomizationProcessor(Processor):
    def __init__(self, renderer, image_paths, num_occlusions, split=pr.TRAIN):
        super(DomainRandomizationProcessor, self).__init__()
        self.copy = pr.Copy()
        self.render = pr.Render(renderer)
        self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        preprocessors = [pr.NormalizeImage()]
        self.preprocess = SequentialProcessor(preprocessors)
        self.split = split

    def call(self):
        input_image, alpha_mask = self.render()
        label_image = self.copy(input_image)

        if self.split == pr.TRAIN:
            input_image = self.augment(input_image, alpha_mask)

        input_image = self.preprocess(input_image)
        label_image = self.preprocess(label_image)
        return input_image, label_image


class DomainRandomization(SequentialProcessor):
    def __init__(self, renderer, size, image_paths,
                 num_occlusions, split=pr.TRAIN):
        super(DomainRandomization, self).__init__()
        self.add(DomainRandomizationProcessor(
            renderer, image_paths, num_occlusions, split))
        self.add(pr.SequenceWrapper(
            {0: {'input_image': [size, size, 3]}},
            {1: {'label_image': [size, size, 3]}}))
