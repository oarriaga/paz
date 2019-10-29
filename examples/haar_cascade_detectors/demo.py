import argparse

from paz.processors import Predict
from paz.processors import ToBoxes2D
from paz.processors import DrawBoxes2D
from paz.processors import ConvertColor
from paz.processors import CastImageToInts
from paz.models import HaarCascadeDetector
from paz.core import SequentialProcessor, VideoPlayer

parser = argparse.ArgumentParser(description='Haarcascade detectors example')
parser.add_argument('-m', '--models', nargs='+', type=str,
                    default=['frontalface_default', 'eye'],
                    help='Model name postfix of openCV')
args = parser.parse_args()

processors = []
for model_name in args.models:
    detector = HaarCascadeDetector(model_name)
    processor = SequentialProcessor()
    preprocessing = [ConvertColor('BGR', 'GRAY')]
    processor.add(Predict(detector, 'image', 'boxes', preprocessing))
    processor.add(ToBoxes2D())
    processor.add(DrawBoxes2D(colors=[0, 255, 0]))
    processor.add(CastImageToInts())
    processors.append(processor)

VideoPlayer((1280, 960), SequentialProcessor(processors)).start()
