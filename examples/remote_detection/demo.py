"""Shows camera detections computed by `serve.py` on another computer.

Open a tunnel to the server, then run the demo with the server's secret:

    ssh -N -L 5000:localhost:5000 <server>
    PAZ_REMOTE_KEY=<secret> python demo.py
"""
import argparse
import os

import paz


def DetectRemotely(connection, quality):
    detect = paz.partial(paz.remote.call, connection, "detect")
    names = ["0"] + paz.datasets.labels("COCO_EFFICIENTDET")
    colors = paz.draw.lincolor(len(names))

    def call(image):
        encoded_image = paz.image.encode_JPEG(image, quality)
        boxes2D = detect(encoded_image)
        return boxes2D, paz.draw.boxes2D(image, *boxes2D, names, colors)

    return call


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    authkey = os.environ["PAZ_REMOTE_KEY"].encode()
    connection = paz.remote.connect(args.port, authkey)
    pipeline = DetectRemotely(connection, args.quality)
    camera = paz.Camera(args.camera)
    paz.VideoPlayer((args.H, args.W), pipeline, camera).run()
