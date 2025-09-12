import jax
import jax.numpy as jp
import numpy as np
import cv2
from paz import SE3

from paz.graphics import PointLight, camera, scene, load, trackball
from paz.graphics.renderer import _render as core_render


def build_viewer(scene_elements, lights):
    """
    Factory function that prepares a scene and returns a JIT-compiled renderer.
    This function is called only once.
    """
    print("Compiling scene and JIT-compiling renderer...")

    flat_scene, processed_lights, mask = scene.compile(
        scene_elements, lights, mask=None
    )

    def render_frame(camera_pose):
        """The lean, fast function that will be JIT-compiled."""
        H, W = 480, 640
        y_FOV = jp.pi / 4.0
        # world_to_camera = jp.linalg.inv(camera_pose)
        # rays = camera.build_rays((H, W), y_FOV, world_to_camera)
        rays = camera.build_rays((H, W), y_FOV, camera_pose)

        image_data, _ = core_render(
            (H, W), camera_pose, rays, flat_scene, processed_lights, mask
        )
        return (jp.clip(image_data, 0, 1) * 255.0).astype(jp.uint8)

    return jax.jit(render_frame)


def main():
    window_name = "Paz Graphics Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    H, W = 480, 640

    print("Setting up the scene...")
    scene_group = load("axes.json")
    print("Parent transform", scene_group.shapes[0].transform)
    print("Parent transform", scene_group.parent_array)

    lights = [
        PointLight(
            intensity=jp.array([1.0, 1.0, 1.0]),
            position=jp.array([-4.0, 5.0, 6.0]),
        )
    ]

    render_function = build_viewer(scene_group, lights)

    initial_camera_pose = SE3.view_transform(
        camera_origin=jp.array([0.0, 10.0, 10.0]),
        target_origin=jp.array([0.0, 0.0, 0.0]),
        world_up=jp.array([0.0, 1.0, 0.0]),
    )

    # -- Trackball State Initialization --
    trackball_state = trackball.TrackballState(
        pose=initial_camera_pose,
        target=jp.zeros(3),
        size=jp.array([W, H]),
        scale=5.0,  # A sensible scale for the scene
    )
    drag_data = None
    mouse_pressed = False
    needs_render = True

    def mouse_callback(event, x, y, flags, param):
        """Handles all mouse events and updates the trackball state."""
        nonlocal trackball_state, drag_data, mouse_pressed, needs_render

        is_ctrl = flags & cv2.EVENT_FLAG_CTRLKEY
        is_shift = flags & cv2.EVENT_FLAG_SHIFTKEY
        mode = trackball.STATE_ROTATE
        if is_ctrl:
            mode = trackball.STATE_ROLL
        elif is_shift:
            mode = trackball.STATE_PAN

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pressed = True
            drag_data = trackball.start_drag(trackball_state, (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
            trackball_state = trackball.drag(drag_data, (x, y), mode)
            needs_render = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            drag_data = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            clicks = 1 if flags > 0 else -1
            trackball_state = trackball.scroll(trackball_state, clicks)
            needs_render = True

    cv2.setMouseCallback(window_name, mouse_callback)

    print(
        "Ready. Use mouse to navigate or h/j/k/l/u/m to move, x/y/z to rotate. Esc to quit."
    )

    while True:
        if needs_render:
            image_data_rgb = render_function(trackball_state.pose)
            needs_render = False

        image_data_bgr = cv2.cvtColor(
            np.array(image_data_rgb), cv2.COLOR_RGB2BGR
        )
        cv2.imshow(window_name, image_data_bgr)

        key = cv2.waitKey(1)
        if key != -1:
            step_size, angle = 0.5, jp.pi / 16
            pose_update = jp.eye(4)

            if key == ord("h"):
                pose_update = SE3.translation([-step_size, 0, 0])
            elif key == ord("l"):
                pose_update = SE3.translation([step_size, 0, 0])
            elif key == ord("j"):
                pose_update = SE3.translation([0, -step_size, 0])
            elif key == ord("k"):
                pose_update = SE3.translation([0, step_size, 0])
            elif key == ord("u"):
                pose_update = SE3.translation([0, 0, step_size])
            elif key == ord("m"):
                pose_update = SE3.translation([0, 0, -step_size])
            elif key == ord("x"):
                pose_update = SE3.rotation_x(angle)
            elif key == ord("y"):
                pose_update = SE3.rotation_y(angle)
            elif key == ord("z"):
                pose_update = SE3.rotation_z(angle)
            elif key == ord("X"):
                pose_update = SE3.rotation_x(-angle)
            elif key == ord("Y"):
                pose_update = SE3.rotation_y(-angle)
            elif key == ord("Z"):
                pose_update = SE3.rotation_z(-angle)
            elif key == 27:
                break

            new_pose = trackball_state.pose @ jp.linalg.inv(pose_update)
            trackball_state = trackball_state._replace(pose=new_pose)
            needs_render = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
