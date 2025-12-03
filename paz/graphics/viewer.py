import jax.numpy as jp
import cv2
import jax
import paz

# from paz.graphics.renderer import _render, _render_with_shadows


def clamp_tilt(tilt, limit=jp.pi / 2 - 0.01):
    return jp.clip(tilt, -limit, limit)


def rotate_camera(delta_x, delta_y, turn, tilt, sensitivity=0.005):
    turn = turn - (sensitivity * delta_x)
    tilt = tilt - (sensitivity * delta_y)
    return {"turn": turn, "tilt": clamp_tilt(tilt)}


def viewer(
    scene,
    camera_pose=jp.eye(4),
    shadows=False,
    light=None,
    H=480,
    W=640,
    y_FOV=0.78,
):
    if light is None:
        light = [paz.graphics.PointLight(jp.ones(3), jp.array([-4, 5, 6]))]

    scene, light, mask = paz.graphics.scene.compile(scene, light, mask=None)
    identity_rays = paz.graphics.camera.build_rays((H, W), y_FOV, jp.eye(4))

    def render(image_shape, camera_pose, rays, scene, light, mask):
        args = (H, W, camera_pose, rays, scene, light, mask, shadows, None, 3)
        return paz.graphics.renderer._render_bounced(*args)

    @jax.jit
    def render_pose(camera_pose):
        """JIT-compiled function for fast, repeated rendering."""
        transform = jp.linalg.inv(camera_pose)
        rays = paz.graphics.geometry.transform_rays(transform, *identity_rays)
        image, depth = render((H, W), camera_pose, rays, scene, light, mask)
        return (jp.clip(image, 0, 1) * 255.0).astype(jp.uint8), depth

    print("Ready. W/A/S/D: Move, Mouse: Look, Esc: Quit.")
    window_name = "PAZ Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

    camera_position = paz.SE3.get_position_vector(camera_pose)
    camera_rotation = {"turn": 0.0, "tilt": 0.0}
    mouse_state = {"last_position": None}

    def mouse_callback(event, x, y, flags, param):
        nonlocal camera_rotation, do_render, mouse_state

        if mouse_state["last_position"] is None:
            mouse_state["last_position"] = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            delta_x = x - mouse_state["last_position"][0]
            delta_y = y - mouse_state["last_position"][1]
            mouse_state["last_position"] = (x, y)
            camera_rotation = rotate_camera(delta_x, delta_y, **camera_rotation)
            do_render = True

    cv2.setMouseCallback(window_name, mouse_callback)
    do_render = True
    while True:
        # if do_render:
        turn_rotation = paz.SE3.rotation_y(camera_rotation["turn"])
        tilt_rotation = paz.SE3.rotation_x(camera_rotation["tilt"])
        rotation = turn_rotation @ tilt_rotation
        camera_pose = rotation.at[:3, 3].set(camera_position)
        (image, _), do_render = paz.time(render_pose)(camera_pose), False
        image = cv2.cvtColor(paz.to_numpy(image), cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key != -1:
            camera_position = key_to_move(key, camera_pose, camera_position)
            # do_render = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def key_to_move(key, camera_pose, position, linear_step=0.05):
    """Calculates a new position vector based on camera orientation."""
    right, up, forward = paz.SE3.get_position_vector(camera_pose)
    if key == ord("d"):
        return position - right * linear_step
    elif key == ord("a"):
        return position + right * linear_step
    elif key == ord("s"):
        return position - forward * linear_step
    elif key == ord("w"):
        return position + forward * linear_step
    return position
