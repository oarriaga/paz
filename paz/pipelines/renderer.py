from ..abstract import SequentialProcessor, Processor
from .. import processors as pr

from .image import AugmentImage


class RenderTwoViews(Processor):
    def __init__(self, renderer):
        super(RenderTwoViews, self).__init__()
        self.render = pr.Render(renderer)

        self.preprocess_image = SequentialProcessor()
        self.preprocess_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image.add(pr.NormalizeImage())

        self.preprocess_alpha = SequentialProcessor()
        self.preprocess_alpha.add(pr.ExpandDims(-1))
        self.preprocess_alpha.add(pr.NormalizeImage())
        self.concatenate = pr.Concatenate(-1)

    def call(self):
        image_A, image_B, matrices, alpha_A, alpha_B = self.render()
        image_A = self.preprocess_image(image_A)
        image_B = self.preprocess_image(image_B)
        alpha_A = self.preprocess_alpha(alpha_A)
        alpha_B = self.preprocess_alpha(alpha_B)
        alpha_channels = self.concatenate([alpha_A, alpha_B])
        return image_A, image_B, matrices, alpha_channels


class RandomizeRenderedImage(SequentialProcessor):
    """Performs alpha blending and data-augmentation to an image and
        it's alpha channel.
    image_paths: List of strings indicating the paths to the images used for
        the background.
    num_occlusions: Int. number of occlusions to be added to the image.
    max_radius_scale: Float between [0, 1] indicating the maximum radius in
        scale of the image size.
    """
    def __init__(self, image_paths, num_occlusions=1, max_radius_scale=0.5):
        super(RandomizeRenderedImage, self).__init__()
        self.add(pr.ConcatenateAlphaMask())
        self.add(pr.BlendRandomCroppedBackground(image_paths))
        for arg in range(num_occlusions):
            self.add(pr.AddOcclusion(max_radius_scale))
        self.add(pr.RandomImageBlur())
        self.add(AugmentImage())
