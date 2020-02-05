from paz.core import SequentialProcessor
from paz import processors as pr
from processors import MeasureSimilarity
from processors import Normalize


class SelfSupervisedAugmentation(SequentialProcessor):
    def __init__(self, renderer, image_paths, size=128,
                 num_occlusions=0, max_radius_scale=0.5, split='train'):

        super(SelfSupervisedAugmentation, self).__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError('Invalid split mode')
        if not isinstance(image_paths, list):
            raise ValueError('``image_paths`` must be list')
        if len(image_paths) == 0:
            raise ValueError('No paths given in ``image_paths``')

        self.renderer = renderer
        self.image_paths = image_paths
        self.split = split
        self.size = size
        self.num_occlusions = num_occlusions
        self.max_radius_scale = max_radius_scale

        self.add(pr.RenderSingleViewSample(self.renderer))
        self.add(pr.ConvertColor('RGB', to='BGR'))
        self.add(pr.Copy('image', 'original_image'))

        if split == 'train':
            self.add(pr.CastImageToFloat())
            self.add(pr.ConcatenateAlphaMask())
            self.add(pr.AddCroppedBackground(self.image_paths, size))
            for occlusion_arg in range(self.num_occlusions):
                self.add(pr.AddOcclusion(self.max_radius_scale))
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomBlur())
            self.add(pr.RandomContrast())
            self.add(pr.RandomBrightness())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('BGR', to='HSV'))
            self.add(pr.CastImageToFloat())
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
            self.add(pr.CastImageToInts())
            self.add(pr.ConvertColor('HSV', to='BGR'))
            self.add(pr.RandomLightingNoise())
        for topic in ['image', 'original_image']:
            self.add(pr.ResizeImage((self.size, self.size), topic))
            self.add(pr.NormalizeImage(topic))
        self.add(pr.OutputSelector(['image'], ['original_image']))

    @property
    def input_shapes(self):
        return [(self.size, self.size, 3)]

    @property
    def label_shapes(self):
        return [(self.size, self.size, 3)]


class AutoEncoderInference(SequentialProcessor):
    """AutoEncoder inference pipeline
    # Arguments
        model: Keras model.
        topic: String. Name of the topic to be outputted.
    # Returns
        Function for outputting reconstructed images
    """
    def __init__(self, model, topic='reconstruction'):
        super(AutoEncoderInference, self).__init__()
        self.topic = topic
        pipeline = [pr.ResizeImage(model.input_shape[1:3]),
                    pr.NormalizeImage(),
                    pr.ExpandDims(axis=0, topic='image')]
        self.add(pr.Predict(model, 'image', self.topic, pipeline))
        self.add(pr.Squeeze(0, self.topic))
        self.add(pr.DenormalizeImage(self.topic))
        self.add(pr.CastImageToInts(self.topic))
        self.add(pr.ShowImage(self.topic, self.topic, False))


class ImplicitRotationInference(SequentialProcessor):
    def __init__(self, encoder, decoder, measure, dictionary):
        super(ImplicitRotationInference, self).__init__()
        decoder_topic, encoder_topic = 'reconstruction', 'latent_vector'
        image_topic, dictionary_topic = 'image', 'dictionary_image'

        # encoder pipeline
        pipeline = [pr.ResizeImage(encoder.input_shape[1:3]),
                    pr.NormalizeImage(),
                    pr.ExpandDims(0, image_topic)]
        self.add(pr.Predict(encoder, image_topic, encoder_topic, pipeline))
        # self.add(pr.Squeeze(0, encoder_topic))
        # self.add(Normalize(encoder_topic))
        self.add(MeasureSimilarity(
            dictionary, measure, encoder_topic, dictionary_topic))
        self.add(pr.ShowImage(dictionary_topic, dictionary_topic, False))

        # decoder pipeline
        self.add(pr.Predict(decoder, encoder_topic, decoder_topic))
        self.add(pr.Squeeze(0, decoder_topic))
        self.add(pr.DenormalizeImage(decoder_topic))
        self.add(pr.CastImageToInts(decoder_topic))
        self.add(pr.ShowImage(decoder_topic, decoder_topic, False))
