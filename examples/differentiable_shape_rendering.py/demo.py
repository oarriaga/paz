import paz
import jax.numpy as jp
import matplotlib.pyplot as plt

camera_origin = jp.array([0.0, 2.0, 4.0])
target_origin = jp.zeros(3)
world_up = jp.array([0, 0, 1])
openGL_to_tamayo = jp.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

left = -2.0
radius = 0.5
y_FOV = jp.pi / 3
H = 480 // 4
W = 640 // 4


world_to_camera_openGL = paz.SE3.view_transform(
    camera_origin, target_origin, world_up
)
world_to_camera_tamayo = openGL_to_tamayo @ world_to_camera_openGL
shape_pose = paz.SE3.rotation_z(jp.pi / 4.0) @ paz.SE3.translation(
    jp.array([left, 0.0, radius])
)

pattern = paz.graphics.Pattern(
    jp.eye(4), paz.graphics.NO_PATTERN, jp.ones((1, 1, 3))
)
material = paz.graphics.Material(jp.full((3,), 0.6), 0.1, 0.1, 0.0, 200.0)
lights = [paz.graphics.PointLight(jp.full((3,), 3.0), camera_origin)]
shape_1 = paz.graphics.Shape(
    paz.SE3.rotation_x(jp.pi / 2.0), paz.graphics.PLANE, pattern, material
)
shape_2 = paz.graphics.Shape(
    shape_pose @ paz.SE3.scaling(jp.array([0.5, 1.0, 0.5])),
    paz.graphics.CUBE,
    pattern,
    material,
)

scene, masks = paz.graphics.shapes.merge(shape_1, shape_2)
rays = paz.graphics.camera.build_rays((H, W), y_FOV, world_to_camera_tamayo)
render = paz.graphics.Render((H, W), world_to_camera_tamayo, rays, True)
pred_image, pred_depth = render(scene, masks, lights)
plt.imshow(pred_image)
plt.show()
