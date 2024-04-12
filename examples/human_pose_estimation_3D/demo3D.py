import argparse
from viz import show3Dpose
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from paz.applications import EstimateHumanPose
from matplotlib.animation import FuncAnimation
from paz.backend.camera import Camera, VideoPlayer
from paz.backend.image import resize_image, show_image


parser = argparse.ArgumentParser(description='Human3D visualization')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

camera = Camera()
camera.intrinsics_from_HFOV(HFOV=70, image_shape=(640, 480))
pipeline = EstimateHumanPose(least_squares, camera.intrinsics)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)


def animate(player):
    """Opens camera and starts continuous inference using ``pipeline``,
       until the user presses ``q`` inside the opened window. Plot the
       3D keypoints on pyplot.
    """
    player.camera.start()
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.manager.set_window_title('Human pose visualization')

    def wrapped_animate(i):
        output = player.step()
        image = resize_image(output[player.topic], tuple(player.image_size))
        show_image(image, 'inference', wait=False)

        keypoints3D = output['keypoints3D']
        if len(keypoints3D) == 0:
            return

        plt.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        show3Dpose(keypoints3D, ax)
    return wrapped_animate


animation = FuncAnimation(plt.gcf(), animate(player), interval=1)
plt.tight_layout()
plt.show()
