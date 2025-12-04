import jax.numpy as jp
import cv2
import jax
import paz
import numpy as np
import time


def clamp_tilt(tilt, limit=jp.pi / 2 - 0.01):
    """Prevents the camera from flipping upside down."""
    return jp.clip(tilt, -limit, limit)


def get_camera_basis(rotation_matrix):
    """
    Extracts the basis vectors from the camera-to-world rotation matrix.
    Assuming standard OpenGL convention where camera looks down -Z.
    """
    # Column 0 is Right (Local X)
    right = rotation_matrix[:3, 0]
    # Column 1 is Up (Local Y)
    up = rotation_matrix[:3, 1]
    # Column 2 is Back (Local Z), so Forward is -Z
    forward = -rotation_matrix[:3, 2]
    return right, up, forward


def viewer(
    scene,
    camera_pose=jp.eye(4),
    shadows=False,
    light=None,
    H=480,
    W=640,
    y_FOV=0.78,
):
    # --- Setup Scene ---
    if light is None:
        light = [
            paz.graphics.PointLight(jp.ones(3), jp.array([-4.0, 5.0, 6.0]))
        ]

    scene, light, mask = paz.graphics.scene.compile(scene, light, mask=None)
    identity_rays = paz.graphics.camera.build_rays((H, W), y_FOV, jp.eye(4))

    # --- Setup Renderer ---
    def render_core(
        image_shape,
        camera_pose_matrix,
        rays,
        scene_data,
        lights_data,
        mask_data,
    ):
        args = (
            H,
            W,
            camera_pose_matrix,
            rays,
            scene_data,
            lights_data,
            mask_data,
            shadows,
            None,
            5,
        )
        return paz.graphics.renderer._render_bounced(*args)

    @jax.jit
    def render_frame(pose_matrix):
        world_to_cam = jp.linalg.inv(pose_matrix)
        rays = paz.graphics.geometry.transform_rays(
            world_to_cam, *identity_rays
        )
        image, depth = render_core(
            (H, W), pose_matrix, rays, scene, light, mask
        )
        return (jp.clip(image, 0, 1) * 255.0).astype(jp.uint8)

    # --- Setup Viewer State ---
    print("------------------------------------------------")
    print("FPS VIEWER (INVERTED CONTROLS):")
    print(" [W, S]       : Move Backward / Forward")
    print(" [A, D]       : Move Right / Left")
    print(" [Q, E]       : Move Up / Down")
    print(" [Shift]      : Move Faster")
    print(" [Mouse]      : Look around")
    print(" [Esc]        : Quit")
    print("------------------------------------------------")

    window_name = "PAZ Raytracer Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, W, H)

    current_pos = np.array(paz.SE3.get_position_vector(camera_pose))

    # SPEEDS
    cam_state = {
        "yaw": 0.0,
        "pitch": 0.0,
        "last_mouse": None,
        "speed_base": 1.0,
        "sensitivity": 0.01,
    }

    # Mouse Callback
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if cam_state["last_mouse"] is None:
                cam_state["last_mouse"] = (x, y)
                return

            dx = x - cam_state["last_mouse"][0]
            dy = y - cam_state["last_mouse"][1]
            cam_state["last_mouse"] = (x, y)

            # Update orientation
            # FLIPPED BACK: Used '-' for standard Mouse Look (Right = Turn Right)
            cam_state["yaw"] -= dx * cam_state["sensitivity"]

            # Pitch
            cam_state["pitch"] -= dy * cam_state["sensitivity"]
            cam_state["pitch"] = clamp_tilt(cam_state["pitch"])

    cv2.setMouseCallback(window_name, mouse_callback)

    # FPS Calculation Init
    prev_time = time.time()

    # --- Main Loop ---
    while True:
        # FPS Logic
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / dt if dt > 0 else 0.0

        rot_y = paz.SE3.rotation_y(cam_state["yaw"])
        rot_x = paz.SE3.rotation_x(cam_state["pitch"])
        rotation_matrix = rot_y @ rot_x

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        right, up, forward = get_camera_basis(rotation_matrix)

        fwd_np = np.array(forward)
        rgt_np = np.array(right)

        current_speed = cam_state["speed_base"]
        move_vec = np.zeros(3)

        # INVERTED KEYBOARD CONTROLS (Maintained per previous request)
        if key == ord("w"):
            move_vec -= fwd_np
        if key == ord("s"):
            move_vec += fwd_np
        if key == ord("d"):
            move_vec -= rgt_np
        if key == ord("a"):
            move_vec += rgt_np
        if key == ord("q"):
            move_vec += np.array([0, 1.0, 0])
        if key == ord("e"):
            move_vec -= np.array([0, 1.0, 0])

        if key == ord("+") or key == ord("="):
            cam_state["speed_base"] *= 1.5
        if key == ord("-") or key == ord("_"):
            cam_state["speed_base"] *= 0.75

        current_pos += move_vec * current_speed
        final_pose = rotation_matrix.at[:3, 3].set(jp.array(current_pos))

        image_jax = render_frame(final_pose)
        image_bgr = cv2.cvtColor(np.array(image_jax), cv2.COLOR_RGB2BGR)

        # Draw Crosshair
        ch_x, ch_y = W // 2, H // 2
        cv2.line(image_bgr, (ch_x - 5, ch_y), (ch_x + 5, ch_y), (0, 255, 0), 1)
        cv2.line(image_bgr, (ch_x, ch_y - 5), (ch_x, ch_y + 5), (0, 255, 0), 1)

        # Draw FPS (Top Right)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            image_bgr,
            fps_text,
            (W - 140, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_name, image_bgr)

    cv2.destroyAllWindows()
