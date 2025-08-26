import jax
import jax.numpy as jp
import numpy as np
import cv2
from paz import SE3

# --- 1. Import Your Graphics Library Modules ---
from paz.graphics.types import PointLight, Material, Shape, Pattern, Group
from paz.graphics.constants import SPHERE, CYLINDER, CONE
from paz.graphics import camera
from paz.graphics import render


# --- 2. JIT-Compiled Render Function ---
@jax.jit
def render_scene(camera_pose, scene, lights):
    """JIT-compiled function to render a scene from a given camera pose."""
    H, W = 480, 640
    y_FOV = jp.pi / 4.0
    world_to_camera = jp.linalg.inv(camera_pose)
    rays = camera.build_rays((H, W), y_FOV, world_to_camera)

    image_data, _ = render(
        image_shape=(H, W),
        world_to_camera=world_to_camera,
        rays=rays,
        shapes=scene,
        lights=lights,
    )
    return (jp.clip(image_data, 0, 1) * 255.0).astype(jp.uint8)


# --- 3. Main Application Setup ---


def main():
    # -- Window Setup --
    window_name = "Paz Graphics Viewer (OpenCV)"
    cv2.namedWindow(window_name)
    running = True

    # -- Build the Scene (once) --
    print("Setting up the scene...")
    scene_group = build_frame_of_reference()
    # flat_scene = prepare_scene(scene_group)
    lights = [
        PointLight(
            intensity=jp.array([1.0, 1.0, 1.0]),
            position=jp.array([-4.0, 5.0, 6.0]),
        )
    ]

    # -- Initialize Camera --
    camera_pose = SE3.view_transform(
        camera_origin=jp.array([2.0, 2.5, 3.0]),
        target_origin=jp.array([0.0, 0.0, 0.0]),
        world_up=jp.array([0.0, 1.0, 0.0]),  # Y-up convention
    )

    # -- Initial Render --
    print("Compiling renderer (this may take a moment)...")
    image_data_rgb = render_scene(camera_pose, scene_group, lights)
    print("Ready.")

    # --- 4. Main Render Loop ---
    while running:
        # -- Display the Image --
        # Convert from JAX array (RGB) to NumPy array (BGR) for OpenCV
        image_data_bgr = cv2.cvtColor(
            np.array(image_data_rgb), cv2.COLOR_RGB2BGR
        )
        cv2.imshow(window_name, image_data_bgr)

        # -- Event Handling --
        # Wait for 1ms for a key press. Returns -1 if no key is pressed.
        key = cv2.waitKey(1)

        if key != -1:  # A key was pressed
            print("Re-rendering...")
            needs_render = True

            # -- Camera Control Logic --
            step_size = 0.2
            angle = jp.pi / 32

            if key == ord("h"):  # Left
                camera_pose @= SE3.translation(jp.array([-step_size, 0, 0]))
            elif key == ord("l"):  # Right
                camera_pose @= SE3.translation(jp.array([step_size, 0, 0]))
            elif key == ord("j"):  # Down
                camera_pose @= SE3.translation(jp.array([0, -step_size, 0]))
            elif key == ord("k"):  # Up
                camera_pose @= SE3.translation(jp.array([0, step_size, 0]))
            elif key == ord("x"):  # Rotate X
                camera_pose @= SE3.rotation_x(angle)
            elif key == ord("y"):  # Rotate Y
                camera_pose @= SE3.rotation_y(angle)
            elif key == ord("z"):  # Rotate Z
                camera_pose @= SE3.rotation_z(angle)
            elif key == 27:  # Escape key
                running = False
            else:
                needs_render = False  # Don't re-render for other keys

            if needs_render:
                image_data_rgb = render_scene(camera_pose, flat_scene, lights)

        # Check if the window was closed by the user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cv2.destroyAllWindows()


def build_frame_of_reference():
    """Helper function to build the frame of reference Group object."""
    grey_material = Material(
        color=jp.array([0.5, 0.5, 0.5]),
        ambient=0.2,
        diffuse=0.8,
        specular=0.2,
        shininess=200.0,
    )
    red_material = Material(
        color=jp.array([1.0, 0.0, 0.0]),
        ambient=0.2,
        diffuse=0.8,
        specular=0.2,
        shininess=200.0,
    )
    green_material = Material(
        color=jp.array([0.0, 1.0, 0.0]),
        ambient=0.2,
        diffuse=0.8,
        specular=0.2,
        shininess=200.0,
    )
    blue_material = Material(
        color=jp.array([0.0, 0.0, 1.0]),
        ambient=0.2,
        diffuse=0.8,
        specular=0.2,
        shininess=200.0,
    )
    default_pattern = Pattern()

    sphere = Shape(
        transform=SE3.scaling(jp.array([0.25, 0.25, 0.25])),
        type=SPHERE,
        material=grey_material,
    )

    cyl_base = SE3.translation(jp.array([0.0, 1.0, 0.0]))
    cyl_scale = SE3.scaling(jp.array([0.05, 1.0, 0.05]))
    cone_tip = SE3.translation(jp.array([0.0, 2.0, 0.0]))
    cone_scale = SE3.scaling(jp.array([0.1, 0.2, 0.1]))

    cyl_x = Shape(
        transform=SE3.rotation_z(-jp.pi / 2.0) @ cyl_base @ cyl_scale,
        type=CYLINDER,
        material=red_material,
    )
    cone_x = Shape(
        transform=SE3.rotation_z(-jp.pi / 2.0) @ cone_tip @ cone_scale,
        type=CONE,
        material=red_material,
    )
    cyl_y = Shape(
        transform=cyl_base @ cyl_scale, type=CYLINDER, material=green_material
    )
    cone_y = Shape(
        transform=cone_tip @ cone_scale, type=CONE, material=green_material
    )
    cyl_z = Shape(
        transform=SE3.rotation_x(jp.pi / 2.0) @ cyl_base @ cyl_scale,
        type=CYLINDER,
        material=blue_material,
    )
    cone_z = Shape(
        transform=SE3.rotation_x(jp.pi / 2.0) @ cone_tip @ cone_scale,
        type=CONE,
        material=blue_material,
    )

    shapes_list = [sphere, cyl_x, cone_x, cyl_y, cone_y, cyl_z, cone_z]
    parent_array = jp.array([-1, 0, 0, 0, 0, 0, 0])
    return Group(shapes=shapes_list, parent_array=parent_array)


if __name__ == "__main__":
    main()
