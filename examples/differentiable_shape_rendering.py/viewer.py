import jax
import jax.numpy as jp
import numpy as np
import cv2
from paz import SE3

from paz.graphics import PointLight, camera, scene, load
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
        world_to_camera = jp.linalg.inv(camera_pose)
        rays = camera.build_rays((H, W), y_FOV, world_to_camera)

        image_data, _ = core_render(
            (H, W), world_to_camera, rays, flat_scene, processed_lights, mask
        )
        return (jp.clip(image_data, 0, 1) * 255.0).astype(jp.uint8)

    return jax.jit(render_frame)


def main():
    window_name = "Paz Graphics Viewer"
    cv2.namedWindow(window_name)
    running = True

    print("Setting up the scene...")
    scene_group = load("axes.json")
    lights = [
        PointLight(
            intensity=jp.array([1.0, 1.0, 1.0]),
            position=jp.array([-4.0, 5.0, 6.0]),
        )
    ]

    render_function = build_viewer(scene_group, lights)

    camera_pose = SE3.view_transform(
        camera_origin=jp.array([2.0, 2.5, 3.0]),
        target_origin=jp.array([0.0, 0.0, 0.0]),
        world_up=jp.array([0.0, 1.0, 0.0]),
    )

    image_data_rgb = render_function(camera_pose)
    print(
        "Ready. h/j/k/l:move, x/y/z:rotate, SHIFT+x/y/z:negative rotate, Esc:quit."
    )

    while running:
        image_data_bgr = cv2.cvtColor(
            np.array(image_data_rgb), cv2.COLOR_RGB2BGR
        )
        cv2.imshow(window_name, image_data_bgr)

        key = cv2.waitKey(1)
        if key != -1:
            needs_render = True
            # FIX: Increased step_size for more responsive movement
            step_size, angle = 0.5, jp.pi / 16

            if key == ord("h"):
                camera_pose @= SE3.translation([-step_size, 0, 0])
            elif key == ord("l"):
                camera_pose @= SE3.translation([step_size, 0, 0])
            elif key == ord("j"):
                camera_pose @= SE3.translation([0, -step_size, 0])
            elif key == ord("k"):
                camera_pose @= SE3.translation([0, step_size, 0])
            elif key == ord("x"):
                camera_pose @= SE3.rotation_x(angle)
            elif key == ord("y"):
                camera_pose @= SE3.rotation_y(angle)
            elif key == ord("z"):
                camera_pose @= SE3.rotation_z(angle)
            # FIX: Added keys for negative rotation (uppercase)
            elif key == ord("X"):
                camera_pose @= SE3.rotation_x(-angle)
            elif key == ord("Y"):
                camera_pose @= SE3.rotation_y(-angle)
            elif key == ord("Z"):
                camera_pose @= SE3.rotation_z(-angle)
            elif key == 27:
                running = False
            else:
                needs_render = False

            if needs_render:
                image_data_rgb = render_function(camera_pose)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
