import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz

parser = argparse.ArgumentParser(description="SSD512 hand detection + pose")
parser.add_argument("--image", default=None, help="RGB image; webcam if unset")
parser.add_argument("-c", "--camera_id", type=int, default=0)
parser.add_argument("--box_scale", default=1.5, type=float)
parser.add_argument("--right_hand", action="store_true")
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--output", default="hand_pose.jpg")
args = parser.parse_args()

estimate = paz.applications.SSD512MinimalHandPose(
    box_scale=args.box_scale, right_hand=args.right_hand)

if args.image is not None:
    inferences, image = estimate(paz.image.load(args.image))
    paz.image.write(args.output, image)
    print(f"wrote {args.output}")
else:
    camera = paz.Camera(identifier=args.camera_id)
    player = paz.VideoPlayer((args.H, args.W), estimate, camera)
    player.run()
