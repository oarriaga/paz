"""Serves an RF-DETR object detector to `demo.py` on another computer.

Both computers must share a secret in the `PAZ_REMOTE_KEY` variable:

    PAZ_REMOTE_KEY=<secret> python serve.py
"""
import argparse
import os

import paz
from paz.applications import DetectRFDETRNano


def DetectEncoded(detect):
    def call(encoded_image):
        return detect(paz.image.decode_JPEG(encoded_image))

    return call


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    args = parser.parse_args()

    authkey = os.environ["PAZ_REMOTE_KEY"].encode()
    detect = DetectEncoded(DetectRFDETRNano(args.score_thresh, draw=False))
    paz.message.info("serving on port", args.port)
    paz.remote.serve({"detect": detect}, args.port, authkey)
