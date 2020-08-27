import argparse

from paz.abstract import Processor
from paz.backend.camera import VideoPlayer, Camera
from paz.models import HaarCascadeDetector
from paz.backend.image import load_image, show_image
import paz.processors as pr


class HaarCascadeDetectors(Processor):
    def __init__(self, model_names):
        super(HaarCascadeDetectors, self).__init__()
        self.model_names = model_names
        self.detectors = []
        for class_arg, model_name in enumerate(self.model_names):
            detector = pr.Predict(
                HaarCascadeDetector(model_name, class_arg),
                pr.ConvertColorSpace(pr.RGB2GRAY),
                pr.ToBoxes2D(args.models))
            self.detectors.append(detector)
        self.draw_boxes2D = pr.DrawBoxes2D(args.models)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = []
        for detector in self.detectors:
            boxes2D.extend(detector(image))
        image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiHaarCascadeDetectors')
    parser.add_argument('-m', '--models', nargs='+', type=str,
                        default=['frontalface_default', 'eye'],
                        help='Model name postfix of openCV xml file')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-p', '--image_path', type=str, default=None,
                        help='full image path used for the pipelines')
    args = parser.parse_args()

    pipeline = HaarCascadeDetectors(args.models)
    if args.image_path is None:
        camera = Camera(args.camera_id)
        player = VideoPlayer((640, 480), pipeline, camera)
        player.run()
    else:
        image = load_image(args.image_path)
        predictions = pipeline(image)
        show_image(predictions['image'])
