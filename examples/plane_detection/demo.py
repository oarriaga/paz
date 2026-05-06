import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
import jax
import jax.numpy as jp
import paz
import matplotlib.pyplot as plt


def to_plane_frame(points, plane_to_world):
    world_to_plane = jp.linalg.inv(plane_to_world)
    homogeneous = jp.concatenate([points, jp.ones((len(points), 1))], axis=1)
    return (world_to_plane @ homogeneous.T).T[:, :3]


def filter_outliers(points, percentile=95):
    dists = jp.linalg.norm(points, axis=1)
    return points[dists <= jp.percentile(dists, percentile)]


def sample(points, max_points):
    n = min(max_points, len(points))
    return points[jp.linspace(0, len(points) - 1, n, dtype=int)]


def mesh_grid(size):
    x, z = jp.meshgrid(
        jp.linspace(-size, size, 20), jp.linspace(-size, size, 20)
    )
    y = jp.zeros_like(x)
    return x, y, z


def draw_points(axis, points):
    axis.scatter(
        points[:, 0],
        points[:, 2],
        points[:, 1],
        c=points[:, 1],
        cmap="viridis",
        s=1,
        alpha=0.6,
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_zlabel("Y")


def draw_mesh(axis, x, y, z):
    axis.plot_surface(x, z, y, alpha=0.3, color="red")


def draw_normal(axis, centroid, normal, length):
    axis.quiver(
        centroid[0],
        centroid[2],
        centroid[1],
        normal[0],
        normal[2],
        normal[1],
        length=length,
        color="red",
        arrow_length_ratio=0.3,
        linewidth=2,
    )


def set_limits(axis, points):
    axis.set_xlim([points[:, 0].min(), points[:, 0].max()])
    axis.set_ylim([points[:, 2].min(), points[:, 2].max()])
    axis.set_zlim([points[:, 1].min(), points[:, 1].max()])


def configure_view(axis):
    axis.view_init(elev=20, azim=45)
    axis.set_box_aspect([1, 1, 1])


def draw_subplot(axis, points, transform, normal, centroid, title):
    plane_points = to_plane_frame(points, transform)
    filtered = filter_outliers(plane_points, percentile=95)
    sampled = sample(filtered, max_points=2000)
    draw_points(axis, sampled)
    draw_mesh(axis, *mesh_grid(size=0.6))
    draw_normal(axis, jp.zeros(3), jp.array([0.0, 1.0, 0.0]), length=0.2)
    set_limits(axis, filtered)
    configure_view(axis)
    axis.set_title(
        f"{title}\nn=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]"
    )


def build_transform(normal, centroid):
    up = jp.array([0.0, 1.0, 0.0])
    y = normal / jp.linalg.norm(normal)
    cross_up_y = jp.cross(up, y)
    x = cross_up_y / jp.linalg.norm(cross_up_y)
    z = jp.cross(y, x)
    rotation = jp.stack([x, y, z], axis=1)
    translation = centroid.reshape(3, 1)
    top = jp.concatenate([rotation, translation], axis=1)
    bottom = jp.array([[0.0, 0.0, 0.0, 1.0]])
    return jp.concatenate([top, bottom], axis=0)


def plot_plane(points, transform, normal, centroid):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_subplot(ax, points, transform, normal, centroid, "Plane Fitting")
    plt.tight_layout()
    plt.show()
    return fig


def plot_comparison(points, normals, centroids, names):
    num_plots = len(normals)
    fig = plt.figure(figsize=(6 * num_plots, 6))
    for i in range(num_plots):
        ax = fig.add_subplot(1, num_plots, i + 1, projection="3d")
        transform = build_transform(normals[i], centroids[i])
        draw_subplot(ax, points, transform, normals[i], centroids[i], names[i])
    plt.tight_layout()
    plt.show()
    return fig


def preprocess_depth(depth, camera_intrinsics, max_depth, transform):
    pointcloud = paz.pointcloud.from_depth(depth, camera_intrinsics)
    pointcloud = paz.algebra.transform_points(transform, pointcloud)
    return paz.pointcloud.bound(pointcloud, max_depth)


def build_plane_graphics_node(plane_to_world):
    floor_material = paz.graphics.Material(jp.full(3, 0.7), 0.1, 0.9, 0.0)
    return paz.graphics.Plane(plane_to_world, floor_material)


shot_arg = 0
class_arg = 1
key = jax.random.PRNGKey(777)
H = 480
W = 640
num_spheres = 100
radius = 0.005
max_depth = 1.0
dataset = "fewsol"
y_FOV = paz.datasets.get_y_FOV(dataset)
camera_intrinsics = paz.datasets.get_intrinsics(dataset)
world_to_camera = jp.eye(4)
size = paz.SE3.scaling(radius)
CV_to_GL = jp.array(
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

pointcloud = preprocess_depth(
    true_depth, camera_intrinsics, max_depth, CV_to_GL
)

# Method 1: Least Squares
normal_ls, offset_ls, centroid_ls = paz.plane.fit_least_squares(pointcloud)

# Method 2: RANSAC
key_ransac, key = jax.random.split(key)
normal_ransac, offset_ransac, inliers = paz.plane.fit_RANSAC(
    key_ransac, pointcloud
)
centroid_ransac = -offset_ransac * normal_ransac

# Method 3: Robust Student-t
key_robust, key = jax.random.split(key)
normal, offset, centroid = paz.plane.fit_student_t(pointcloud)

# Method 4: Robust Laplace
normal_laplace, offset_laplace, centroid_laplace = paz.plane.fit_laplace(
    pointcloud
)

# Visualize comparison of all four methods
normals = [normal_ls, normal_ransac, normal, normal_laplace]
centroids = [centroid_ls, centroid_ransac, centroid, centroid_laplace]
names = ["Least Squares", "RANSAC", "Robust Student-t", "Robust Laplace"]
plot_comparison(pointcloud, normals, centroids, names)

# Use the robust method for rendering
world_up = jax.nn.one_hot(1, 3)
plane_to_world = paz.plane.build_plane_to_world(world_up, normal, centroid)
pointcloud_filtered = paz.pointcloud.filter_above_plane(
    pointcloud, plane_to_world, 0.01
)

nodes = [build_plane_graphics_node(plane_to_world)]
for position in paz.pointcloud.sample(key, pointcloud_filtered, num_spheres):
    nodes.append(paz.graphics.Sphere(paz.SE3.translation(position) @ size))
scene = paz.graphics.Scene(nodes)

lights = [paz.graphics.PointLight(jp.ones(3), jp.ones(3))]
render_args = (H, W), y_FOV, world_to_camera
render_karg = dict(
    lights=lights,
    mask=None,
    shadows=False,
    tiles=(1, 1),
    chunk_size=1024,
)
render = jax.jit(paz.partial(paz.graphics.render, *render_args, **render_karg))
pred_image, pred_depth = render(scene=scene)

pred_image = paz.image.denormalize(pred_image)
true_image = (
    true_image if dataset == "fewsol" else paz.image.denormalize(true_image)
)
mosaic = paz.draw.mosaic(jp.array([true_image, pred_image]), (1, 2))
paz.image.show(mosaic.astype(jp.uint8))
