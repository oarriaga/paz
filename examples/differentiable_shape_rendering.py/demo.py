import matplotlib.pyplot as plt
import trimesh
import pyrender
import jax.numpy as jp
import paz
from paz import SE3

H = 480 // 5
W = 640 // 5
y_FOV = jp.pi / 3
box_size = 100.0
left = -2.0
radius = 0.5
shape_pose = SE3.rotation_z(jp.pi / 4.0) @ SE3.translation(
    jp.array([left, 0.0, radius])
)

camera_origin = jp.array([0.0, 2.0, 4.0])
target_origin = jp.zeros(3)
world_up = jp.array([0, 0, 1])
world_to_camera_openGL = SE3.view_transform(
    camera_origin, target_origin, world_up
)
openGL_to_tamayo = jp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
world_to_camera_tamayo = openGL_to_tamayo @ world_to_camera_openGL
camera_openGL_to_world = jp.linalg.inv(world_to_camera_openGL)

# build pyrender scene
floor = trimesh.creation.box(extents=[box_size, box_size, 0.1])
shape = trimesh.creation.box([1.0, 2.0, 1.0])
floor = pyrender.Mesh.from_trimesh(floor)
shape = pyrender.Mesh.from_trimesh(shape)
scene = pyrender.Scene()
floor_node = scene.add(floor)
shape_node = scene.add(shape, pose=shape_pose)
light = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=100.0)
camera = pyrender.PerspectiveCamera(yfov=y_FOV, aspectRatio=W / H, znear=0.001)
scene.add(camera, pose=camera_openGL_to_world)
scene.add(light, pose=camera_openGL_to_world)
renderer = pyrender.OffscreenRenderer(W, H)
true_image, true_depth = renderer.render(scene)

# build tamayo scene
pattern = paz.graphics.Pattern(
    jp.eye(4), paz.graphics.NO_PATTERN, jp.ones((1, 1, 3))
)
material = paz.graphics.Material(jp.full((3,), 0.6), 0.1, 0.1, 0.0, 200.0)
lights = [paz.graphics.PointLight(jp.full((3,), 3.0), camera_origin)]
shape_1 = paz.graphics.Shape(
    SE3.rotation_x(jp.pi / 2.0), paz.graphics.PLANE, pattern, material
)
shape_2 = paz.graphics.Shape(
    shape_pose @ SE3.scaling(jp.array([0.5, 1.0, 0.5])),
    paz.graphics.CUBE,
    pattern,
    material,
)
scene_tamayo = paz.graphics.shapes.merge(shape_1, shape_2)
rays = paz.graphics.camera.build_rays((H, W), y_FOV, world_to_camera_tamayo)
render = paz.graphics.Render((H, W), world_to_camera_tamayo, rays, False)
pred_image, pred_depth = render(scene_tamayo, jp.ones(2), lights)
pred_image = (255.0 * pred_image).astype("uint8")
true_pred_image = jp.concatenate([true_image, pred_image], axis=1)
plt.imshow(true_pred_image)
plt.show()

plt.imshow(true_depth)
plt.colorbar()
plt.show()

plt.imshow(pred_depth)
plt.colorbar()
plt.show()

plt.imshow(jp.abs(true_depth - pred_depth))
plt.colorbar()
plt.show()

true_pred_depth = jp.concatenate([true_depth, pred_depth], axis=1)
plt.title("Left: depth from renderer A,        Right: depth from renderer B")
plt.imshow(true_pred_depth, cmap="PiYG")
plt.colorbar()
plt.show()

paz.assert_snapshot(pred_depth, "pred_depth.npy")
paz.assert_snapshot(pred_image, "pred_image.npy")
