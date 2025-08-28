import jax.numpy as jp
import numpy as np
import cv2
import jax
import paz
from paz.graphics.renderer import _render as render


def viewer(
    scene, camera_pose=jp.eye(4), light=None, image_size=(480, 640), y_FOV=0.78
):
    if light is None:
        light = [paz.graphics.PointLight(jp.ones(3), jp.array([-4, 5, 6]))]
    scene, light, mask = paz.graphics.scene.compile(scene, light, mask=None)
    identity_rays = paz.graphics.camera.build_rays(image_size, y_FOV, jp.eye(4))

    def _render_pose(camera_pose):
        inv = jp.linalg.inv(camera_pose)
        rays = paz.graphics.geometry.transform_rays(inv, *identity_rays)
        image, depth = render(image_size, camera_pose, rays, scene, light, mask)
        image = paz.image.denormalize(jp.clip(image, 0, 1))
        return image, depth

    render_pose, do_render = jax.jit(_render_pose), True
    print("Navigate h/j/k/l/u/m to move, x/y/z to rotate. Esc to quit.")
    window_name = "PAZ-VIEWER"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    while True:
        if do_render:
            print("rendering...")
            (image, _), do_render = render_pose(camera_pose), False
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, image)

        keystroke = cv2.waitKey(1)
        if keystroke == 27:
            break
        elif keystroke != -1:
            camera_pose = camera_pose @ keystroke_to_transform(keystroke)
            do_render = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def keystroke_to_transform(keystroke, linear_step=0.5, angular_step=jp.pi / 16):
    if keystroke == ord("h"):
        pose_update = paz.SE3.translation([linear_step, 0, 0])
    elif keystroke == ord("l"):
        pose_update = paz.SE3.translation([-linear_step, 0, 0])
    elif keystroke == ord("j"):
        pose_update = paz.SE3.translation([0, linear_step, 0])
    elif keystroke == ord("k"):
        pose_update = paz.SE3.translation([0, -linear_step, 0])
    elif keystroke == ord("u"):
        pose_update = paz.SE3.translation([0, 0, -linear_step])
    elif keystroke == ord("m"):
        pose_update = paz.SE3.translation([0, 0, linear_step])
    elif keystroke == ord("x"):
        pose_update = paz.SE3.rotation_x(-angular_step)
    elif keystroke == ord("y"):
        pose_update = paz.SE3.rotation_y(-angular_step)
    elif keystroke == ord("z"):
        pose_update = paz.SE3.rotation_z(-angular_step)
    elif keystroke == ord("a"):
        pose_update = paz.SE3.rotation_x(angular_step)
    elif keystroke == ord("b"):
        pose_update = paz.SE3.rotation_y(angular_step)
    elif keystroke == ord("g"):
        pose_update = paz.SE3.rotation_z(angular_step)
    else:
        return jp.eye(4)
    return pose_update
