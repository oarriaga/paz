import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from keras.utils import get_file
import paz

URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.14/image_with_hand.png"  # fmt: skip

parser = argparse.ArgumentParser(description="Minimal hand pose on an image")
parser.add_argument("--image", default=None, help="RGB image; test image if unset")  # fmt: skip
parser.add_argument("--right_hand", action="store_true")
parser.add_argument("--output", default="hand_pose.jpg")
args = parser.parse_args()

path = args.image or get_file(os.path.basename(URL), URL, cache_subdir="paz/tests")  # fmt: skip
estimate = paz.applications.MinimalHandPoseEstimation(right_hand=args.right_hand)
hand, image = estimate(paz.image.load(path))
paz.image.write(args.output, image)
print(f"wrote {args.output}")
