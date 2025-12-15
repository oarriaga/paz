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


def preprocess(depth, camera_intrinsics, max_depth, num_points, transform):
    pointcloud = paz.pointcloud.from_depth(depth, camera_intrinsics)
    pointcloud = paz.algebra.transform_points(transform, pointcloud)
    return paz.pointcloud.bound(pointcloud, max_depth)


def predict(key, x, step_size, scale, DOF):
    loss = paz.lock(paz.plane.student_t_loss, scale, DOF, x)
    optimize = paz.lock(paz.plane.fit, optax.adam(step_size), loss, num_steps)
    (pred_normal, pred_centroid), losses = optimize(key, x)
    return pred_normal, pred_centroid


pointcloud = preprocess(true_depth, camera_intrinsics, max_depth, num_points,openCV_to_openGL)  # fmt: skip
samples = paz.pointcloud.sample(key, pointcloud, num_points)
pred_normal, pred_centroid = predict(key, samples, step_size, scale, DOF)

camera_pose = jp.eye(4)
world_up = jp.array([0.0, 1.0, 0.0])
y_axis = pred_normal / jp.linalg.norm(pred_normal)

x_axis = jp.cross(world_up, y_axis)
x_axis = x_axis / jp.linalg.norm(x_axis)
z_axis = jp.cross(y_axis, x_axis)
rotation = jp.stack([x_axis, y_axis, z_axis], axis=1)

plane_pose = paz.SE3.to_affine_matrix(rotation, pred_centroid)

pointcloud_plane = paz.algebra.transform_points(jp.linalg.inv(plane_pose), pointcloud)  # fmt: skip
mask = pointcloud_plane[:, 1] > 0.01

floor_material = paz.graphics.Material(jp.full(3, 0.7), 0.1, 0.9, 0.0)
nodes = [paz.graphics.Plane(plane_pose, floor_material)]
positions = paz.pointcloud.sample(key, pointcloud[mask], num_spheres)
scale = paz.SE3.scaling(radius)
for position in positions:
    nodes.append(paz.graphics.Sphere(paz.SE3.translation(position) @ scale))
scene = paz.graphics.Scene(nodes)

lights = [paz.graphics.PointLight(jp.ones(3), jp.ones(3))]
rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
render_args = ((H, W), camera_pose, rays)
render_karg = {"lights": lights, "mask": None, "shadows": False}
render = jax.jit(paz.partial(paz.graphics.render, *render_args, **render_karg))
pred_image, pred_depth = render(scene=scene)
pred_image = paz.image.denormalize(pred_image)
true_image = true_image if dataset == "fewsol" else paz.image.denormalize(true_image)  # fmt: skip
mosaic = paz.draw.mosaic(jp.array([true_image, pred_image]), (1, 2))
paz.image.show(mosaic.astype(jp.uint8))
