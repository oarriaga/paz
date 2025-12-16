import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import jax
import paz
import jax.numpy as jp
import optax


key = jax.random.PRNGKey(777)
step_size = 0.1
dtype = jp.bfloat16
num_steps = 100
DOF = 2.0
scale = 0.3
shot_arg = 0
num_points = 100
H = 480
W = 640
num_spheres = 100
radius = 0.005
class_arg = 3
shot_arg = 0
max_depth = 1.0
dataset = "fewsol"
y_FOV = paz.datasets.get_y_FOV(dataset)
camera_intrinsics = paz.datasets.get_intrinsics(dataset)
world_to_camera = jp.eye(4)
size = paz.SE3.scaling(radius)
openCV_to_openGL = jp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


if dataset == "fewsol":
    shot = paz.datasets.fewsol.load(class_arg)[shot_arg]
    true_image, true_depth = shot.image.copy(), shot.depth.copy()
elif dataset == "fsclvr":
    dataset = paz.datasets.fsclvr.load("plain", "train")
    true_image, true_depth, label = dataset[class_arg][shot_arg]
else:
    raise ValueError


def preprocess_depth(depth, camera_intrinsics, max_depth, transform):
    pointcloud = paz.pointcloud.from_depth(depth, camera_intrinsics)
    pointcloud = paz.algebra.transform_points(transform, pointcloud)
    return paz.pointcloud.bound(pointcloud, max_depth)


def predict(key, x, step_size, scale, DOF):
    loss = paz.lock(paz.plane.student_t_loss, scale, DOF, x)
    optimize = paz.lock(paz.plane.fit, optax.adam(step_size), loss, num_steps)
    (pred_normal, pred_centroid), losses = optimize(key, x)
    return pred_normal, pred_centroid


def build_plane_to_world(world_up, normal, position):
    y_axis = normal / jp.linalg.norm(normal)
    x_axis = jp.cross(world_up, y_axis)
    x_axis = x_axis / jp.linalg.norm(x_axis)
    z_axis = jp.cross(y_axis, x_axis)
    rotation = jp.stack([x_axis, y_axis, z_axis], axis=1)
    return paz.SE3.to_affine_matrix(rotation, position)


def filter_above_height(plane_to_world, pointcloud, height=0.01):
    world_to_plane = jp.linalg.inv(plane_to_world)
    pointcloud_plane = paz.algebra.transform_points(world_to_plane, pointcloud)
    mask = pointcloud_plane[:, 1] > height
    return pointcloud[mask]


def build_plane(plane_to_world):
    floor_material = paz.graphics.Material(jp.full(3, 0.7), 0.1, 0.9, 0.0)
    return paz.graphics.Plane(plane_to_world, floor_material)


pointcloud = preprocess_depth(true_depth, camera_intrinsics, max_depth, openCV_to_openGL)  # fmt: skip
samples = paz.pointcloud.sample(key, pointcloud, num_points)
normal, centroid = predict(key, samples, step_size, scale, DOF)
plane_to_world = build_plane_to_world(jax.nn.one_hot(1, 3), normal, centroid)
pointcloud_filtered = filter_above_height(plane_to_world, pointcloud, 0.01)

nodes = [build_plane(plane_to_world)]
for position in paz.pointcloud.sample(key, pointcloud_filtered, num_spheres):
    nodes.append(paz.graphics.Sphere(paz.SE3.translation(position) @ size))
scene = paz.graphics.Scene(nodes)

lights = [paz.graphics.PointLight(jp.ones(3), jp.ones(3))]
rays = paz.graphics.camera.build_rays((H, W), y_FOV, world_to_camera)
render_args = ((H, W), world_to_camera, rays)
render_karg = {"lights": lights, "mask": None, "shadows": False}
render = jax.jit(paz.partial(paz.graphics.render, *render_args, **render_karg))
pred_image, pred_depth = render(scene=scene)

pred_image = paz.image.denormalize(pred_image)
true_image = true_image if dataset == "fewsol" else paz.image.denormalize(true_image)  # fmt: skip
mosaic = paz.draw.mosaic(jp.array([true_image, pred_image]), (1, 2))
paz.image.show(mosaic.astype(jp.uint8))
