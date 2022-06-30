import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.applications import MinimalHandPoseEstimation
from paz.backend.image import resize_image, show_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from paz.datasets import MINIMAL_HAND_CONFIG
from paz.backend.image import plot_3D_keypoints_link


parser = argparse.ArgumentParser(description='Minimal hand keypoint detection')
parser.add_argument('-c', '--camera_id', type=int, default=0,
                    help='Camera device ID')
args = parser.parse_args()

pipeline = MinimalHandPoseEstimation(right_hand=False)
camera = Camera(args.camera_id)
player = VideoPlayer((640, 480), pipeline, camera)


link_orders = MINIMAL_HAND_CONFIG['part_orders']
link_args = MINIMAL_HAND_CONFIG['part_arg']
link_colors = MINIMAL_HAND_CONFIG['part_color']
link_colors = np.array(link_colors)/255
joint_colors = MINIMAL_HAND_CONFIG['joint_color']
joint_colors = np.array(joint_colors)/255

def animate(player):
    """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window. Plot the
        3D keypoints on pyplot.
    """  
    player.camera.start()
    ax = plt.axes(projection='3d')
    ax.view_init(-160, -80)
    ax.figure.canvas.set_window_title('Minimal hand 3D plot')

    def wrapped_animate(i):
        output = player.step()
        image = resize_image(output[player.topic], tuple(player.image_size))
        show_image(image, 'inference', wait=False)
        
        keypoints3D = output['keypoints3D']
        xs, ys, zs = np.split(keypoints3D, 3, axis=1)
           
        plt.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter3D(xs, ys, zs, c = joint_colors)
        plot_3D_keypoints_link(ax, keypoints3D, link_args, link_orders,
                               link_colors)
    return wrapped_animate        
        

animation = FuncAnimation(plt.gcf(), animate(player), interval=1)
plt.tight_layout()
plt.show()


            