import math
import jax.numpy as jp
import cv2
import jax
import paz
import numpy as np
import time
import json
import datetime


def clamp_tilt(tilt, limit=None):
    if limit is None:
        limit = math.pi / 2 - 0.01
    return jp.clip(tilt, -limit, limit)


def get_camera_basis(rotation_matrix):
    right = rotation_matrix[:3, 0]
    up = rotation_matrix[:3, 1]
    forward = -rotation_matrix[:3, 2]
    return right, up, forward


def _default_lights():
    position = jp.array([-4.0, 5.0, 6.0])
    return [paz.graphics.PointLight(jp.ones(3), position)]


def _build_identity_rays(H, W, y_FOV):
    args = ((H, W), y_FOV, jp.eye(4))
    return paz.graphics.camera.build_rays(*args)


def _transform_rays(pose_matrix, identity_rays):
    cam_to_world = jp.linalg.inv(pose_matrix)
    transform = paz.graphics.geometry.transform_rays
    return transform(cam_to_world, *identity_rays)


def _to_uint8(image):
    return (jp.clip(image, 0, 1) * 255.0).astype(jp.uint8)


def shape_renderer(scene, H, W, y_FOV=0.78, lights=None, shadows=False):
    if lights is None:
        lights = _default_lights()
    compiled = paz.graphics.scene.compile(scene, lights, mask=None)
    scene, mask, _, lights = compiled
    identity_rays = _build_identity_rays(H, W, y_FOV)
    bounces = paz.graphics.scene.compute_bounces(scene)

    @jax.jit
    def render_frame(pose_matrix):
        rays = _transform_rays(pose_matrix, identity_rays)
        render_bounced = paz.graphics.renderer.render_bounced
        args = (H, W, pose_matrix, rays, scene, lights, mask)
        image, _ = render_bounced(*args, shadows, None, bounces)
        return _to_uint8(image)

    return render_frame


def mesh_renderer(meshes, mask, H, W, y_FOV=0.78, lights=None, chunk_size=1024):
    if lights is None:
        lights = _default_lights()
    identity_rays = _build_identity_rays(H, W, y_FOV)

    @jax.jit
    def render_frame(pose_matrix):
        rays = _transform_rays(pose_matrix, identity_rays)
        args = ((H, W), pose_matrix, rays, meshes, mask, lights, chunk_size)
        image, _ = paz.graphics.mesh.render(*args)
        return _to_uint8(image)

    return render_frame


def viewer(render_fn_or_scene, camera_pose=None, shadows=False, light=None, H=480, W=640, y_FOV=0.78):
    if callable(render_fn_or_scene):
        render_fn = render_fn_or_scene
    else:
        if light is None:
            light = _default_lights()
        args = (render_fn_or_scene, H, W, y_FOV, light, shadows)
        render_fn = shape_renderer(*args)
    _run_viewer(render_fn, camera_pose, H, W)


def _run_viewer(render_fn, camera_pose, H, W):
    if camera_pose is None:
        camera_pose = jp.eye(4)

    _print_controls()
    window_name = "PAZ Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, W, H)

    current_pos = np.array(
        paz.SE3.get_position_vector(camera_pose))
    cam_state = {
        "yaw": 0.0,
        "pitch": 0.0,
        "last_mouse": None,
        "speed_base": 1.0,
        "sensitivity": 0.01,
    }
    callback = _make_mouse_callback(cam_state)
    cv2.setMouseCallback(window_name, callback)
    prev_time = time.time()

    while True:
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / dt if dt > 0 else 0.0

        rot_y = paz.SE3.rotation_y(cam_state["yaw"])
        rot_x = paz.SE3.rotation_x(cam_state["pitch"])
        rotation = rot_y @ rot_x

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        prop = cv2.getWindowProperty(
            window_name, cv2.WND_PROP_VISIBLE)
        if prop < 1:
            break

        move = _compute_movement(key, rotation, cam_state)
        current_pos += move
        pose = rotation.at[:3, 3].set(jp.array(current_pos))

        image_jax = render_fn(pose)
        image_bgr = cv2.cvtColor(
            np.array(image_jax), cv2.COLOR_RGB2BGR)

        if key == ord("c"):
            _save_camera_pose(pose)
        if key == ord("p"):
            _save_screenshot(image_jax)

        _draw_overlay(image_bgr, fps, W, H)
        cv2.imshow(window_name, image_bgr)

    cv2.destroyAllWindows()


def _print_controls():
    print("------------------------------------------------")
    print("PAZ VIEWER (FPS CONTROLS):")
    print(" [W, S]       : Move Backward / Forward")
    print(" [A, D]       : Move Right / Left")
    print(" [Q, E]       : Move Up / Down")
    print(" [+, -]       : Speed Up / Slow Down")
    print(" [C]          : Save Camera Pose")
    print(" [P]          : Take Screenshot")
    print(" [Mouse]      : Look around")
    print(" [Esc]        : Quit")
    print("------------------------------------------------")


def _make_mouse_callback(cam_state):
    def callback(event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        if cam_state["last_mouse"] is None:
            cam_state["last_mouse"] = (x, y)
            return
        dx = x - cam_state["last_mouse"][0]
        dy = y - cam_state["last_mouse"][1]
        cam_state["last_mouse"] = (x, y)
        sens = cam_state["sensitivity"]
        cam_state["yaw"] -= dx * sens
        cam_state["pitch"] -= dy * sens
        cam_state["pitch"] = clamp_tilt(cam_state["pitch"])
    return callback


def _compute_movement(key, rotation, cam_state):
    right, up, forward = get_camera_basis(rotation)
    fwd = np.array(forward)
    rgt = np.array(right)
    move = np.zeros(3)
    if key == ord("w"):
        move -= fwd
    if key == ord("s"):
        move += fwd
    if key == ord("d"):
        move -= rgt
    if key == ord("a"):
        move += rgt
    if key == ord("q"):
        move += np.array([0, 1.0, 0])
    if key == ord("e"):
        move -= np.array([0, 1.0, 0])
    if key == ord("+") or key == ord("="):
        cam_state["speed_base"] *= 1.5
    if key == ord("-") or key == ord("_"):
        cam_state["speed_base"] *= 0.75
    return move * cam_state["speed_base"]


def _save_camera_pose(pose):
    stamp = _timestamp()
    pose_np = np.array(pose)
    print(f"Camera Pose at {stamp}:")
    print(pose_np)
    filename = f"camera_pose_{stamp}.json"
    with open(filename, "w") as f:
        json.dump(pose_np.tolist(), f, indent=4)
    print(f"Saved camera pose to {filename}")


def _save_screenshot(image):
    stamp = _timestamp()
    filename = f"capture_{stamp}.png"
    paz.image.write(filename, image)
    print(f"Saved screenshot to {filename}")


def _timestamp():
    fmt = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(fmt)


def _draw_overlay(image_bgr, fps, W, H):
    cx, cy = W // 2, H // 2
    color = (0, 255, 0)
    cv2.line(image_bgr, (cx - 5, cy), (cx + 5, cy), color, 1)
    cv2.line(image_bgr, (cx, cy - 5), (cx, cy + 5), color, 1)
    fps_text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (W - 140, 30)
    cv2.putText(image_bgr, fps_text, pos, font, 0.7, color, 2)
