import argparse
import paz
from paz.models.detection import HaarCascadeDetector

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--image_path", default=0, type=int)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--models", nargs=2, default=["frontalface_default", "eye"])
args = parser.parse_args()


def Detector(labels, colors):
    draw_functions = [paz.lock(paz.draw.boxes, color, 2) for color in colors]

    detectors = []
    for class_arg, label in enumerate(labels):
        detectors.append(HaarCascadeDetector(label, 1.3, 5, class_arg))

    def call(image):
        boxes, image_with_boxes = [], image.copy()
        for detect, draw in zip(detectors, draw_functions):
            class_boxes = detect(image).boxes
            image_with_boxes = draw(image_with_boxes, class_boxes)
            boxes.append(class_boxes)
        boxes = paz.boxes.join(boxes)
        return paz.message.Detections(image_with_boxes, boxes)

    return call


colors = paz.draw.lincolor(len(args.models), normalize=False)
pipeline = Detector(args.models, colors)
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
