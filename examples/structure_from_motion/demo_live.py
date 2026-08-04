import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import shutil
import cv2
import jax
import matplotlib.pyplot as plt
import paz

import backend
import pipeline

parser = argparse.ArgumentParser(description="Structure from motion (webcam)")
parser.add_argument("--camera_id", type=int, default=0)
parser.add_argument("--HFOV", type=float, default=70,
                    help="Horizontal field of view in degrees")
parser.add_argument("--match_ratio", type=float, default=0.75)
parser.add_argument("--residual_thresh", type=float, default=0.5,
                    help="Sampson-distance RANSAC threshold between frames")
parser.add_argument("--correspondence_thresh", type=float, default=0.5,
                    help="Sampson-distance RANSAC threshold for PnP tracking")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--frame_skip", type=int, default=20,
                    help="Keep one out of every this many recorded frames")
args = parser.parse_args()

IMAGES_PATH = "./images"


def plot_3D_keypoints(points3D_list, colors_list, outlier_thresh=80):
    axis = plt.axes(projection="3d")
    axis.view_init(-160, -80)
    axis.set_xlabel("X"), axis.set_ylabel("Y"), axis.set_zlabel("Z")
    for points3D, colors in zip(points3D_list, colors_list):
        points3D, inliers = backend.remove_outliers(points3D, outlier_thresh)
        axis.scatter(*points3D.T, s=5, c=colors[inliers] / 255.0)
    plt.show()


def record_frames(camera, images_path, frame_skip):
    """Records a sequence of frames while the camera moves around the scene.

    paz.VideoPlayer.record_frames/extract_frames_from_video are currently
    broken on this branch (they call resize_image/show_image/convert_
    color_space, which no longer exist), so this records straight to disk
    with the same paz.Camera/paz.image primitives paz.VideoPlayer.run() uses.
    """
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.makedirs(images_path)
    camera.start()
    print("Press 'q' in the capture window to stop recording.")
    frame_arg = 0
    while True:
        image = camera.read()
        if image is None:
            continue
        paz.image.show(image, "recording", wait=False)
        if frame_arg % frame_skip == 0:
            frame_name = f"{frame_arg // frame_skip:03d}.png"
            paz.image.write(os.path.join(images_path, frame_name), image)
        frame_arg += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.stop()
    cv2.destroyAllWindows()


camera = paz.Camera(identifier=args.camera_id)
camera_intrinsics = paz.to_numpy(camera.intrinsics_from_HFOV(args.HFOV))

answer = input("Start recording? yes/no: ")
if answer == "yes":
    record_frames(camera, IMAGES_PATH, args.frame_skip)

image_names = sorted(os.listdir(IMAGES_PATH))
images = [paz.to_numpy(paz.image.load(os.path.join(IMAGES_PATH, name)))
         for name in image_names]

key = jax.random.PRNGKey(args.seed)
reconstruction = pipeline.reconstruct_scene(
    images, camera_intrinsics, key, args.match_ratio, args.residual_thresh,
    args.correspondence_thresh)
plot_3D_keypoints(reconstruction.points3D, reconstruction.colors)
