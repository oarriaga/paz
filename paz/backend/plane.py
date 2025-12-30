import jax
import paz
import jax.numpy as jp
import optax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def fit_least_squares(pointcloud):
    """Fit plane using least squares SVD method."""
    centroid = jp.mean(pointcloud, axis=0)
    centered = pointcloud - centroid
    U, S, Vt = jp.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    normal = jp.where(normal[1] < 0, -normal, normal)
    offset = -jp.dot(normal, centroid)
    return normal, offset, centroid


def fit_RANSAC(key, pointcloud, state=None, steps=100, threshold=0.02):
    """Fit plane using RANSAC with least squares refinement."""

    def initialize_state(pointcloud):
        normal, offset, count = jp.array([0.0, 1.0, 0.0]), 0.0, 0
        mask = jp.zeros(len(pointcloud), dtype=bool)
        return (normal, offset, count, mask)

    def find_inliers(pointcloud, normal, offset, threshold):
        distances = jp.abs(jp.dot(pointcloud, normal) + offset)
        mask = distances < threshold
        count = jp.sum(mask)
        return mask, count

    def compute_normal_from_three_points(points):
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = jp.cross(v1, v2)
        length = jp.linalg.norm(normal)
        world_up = jp.array([0.0, 1.0, 0.0])
        normal = jp.where(length > 1e-6, normal / length, world_up)
        normal = jp.where(normal[1] < 0, -normal, normal)
        return normal

    def step(state, key):
        best_normal, best_offset, best_count, best_mask = state
        samples = paz.pointcloud.sample(key, pointcloud, num_points=3)
        normal = compute_normal_from_three_points(samples)
        offset = -jp.dot(normal, samples[0])
        mask, count = find_inliers(pointcloud, normal, offset, threshold)
        should_update = count > best_count

        def update():
            return (normal, offset, count, mask)

        def keep():
            return state

        new_state = jax.lax.cond(should_update, update, keep)
        return new_state, None

    def find_best_parameters(state):
        normal, offset, count, mask = state
        return normal, offset, mask

    def refine():
        return fit_least_squares(inlier_points)

    def keep():
        return (normal, offset, inlier_points.mean(axis=0))

    state = initialize_state(pointcloud) if state is None else state
    keys = jax.random.split(key, steps)
    final_state, _ = jax.lax.scan(step, state, keys)
    normal, offset, inliers = find_best_parameters(final_state)
    inlier_points = pointcloud[inliers]
    has_enough_inliers = len(inlier_points) > 3
    normal, offset, _ = jax.lax.cond(has_enough_inliers, refine, keep)
    return normal, offset, inliers


def optimize(parameters, optimizer, compute_loss, steps):
    compute_gradients = jax.value_and_grad(compute_loss, [0, 1])

    def step(state, _):
        parameters, opt_state = state
        loss_val, grads = compute_gradients(*parameters)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return (parameters, opt_state), loss_val

    opt_state = optimizer.init(parameters)
    state = (parameters, opt_state)
    final_state, losses = jax.lax.scan(step, state, jp.arange(steps))
    (normal, centroid), _ = final_state
    normal = normal / jp.linalg.norm(normal)
    return (normal, centroid), losses


def fit_student_t(
    pointcloud, state=None, optimizer=None, scale=0.01, DOF=1.5, steps=200
):

    def compute_student_t_loss(normal, centroid, scale, DOF, points):
        unit_normal = normal / jp.linalg.norm(normal)
        distances = jp.dot(points - centroid, unit_normal)
        return -tfd.StudentT(DOF, 0.0, scale).log_prob(distances).sum()

    def initialize_state(pointcloud):
        return (jp.array([0.0, 1.0, 0.0]), jp.mean(pointcloud, axis=0))

    state = initialize_state(pointcloud) if state is None else state
    optimizer = optax.adam(0.05) if optimizer is None else optimizer
    loss = paz.lock(compute_student_t_loss, scale, DOF, pointcloud)
    (normal, centroid), losses = optimize(state, optimizer, loss, steps)
    offset = -jp.dot(normal, centroid)
    return normal, offset, centroid


def fit_laplace(pointcloud, state=None, optimizer=None, scale=0.01, steps=200):

    def compute_laplace_loss(normal, centroid, scale, points):
        unit_normal = normal / jp.linalg.norm(normal)
        distances = jp.dot(points - centroid, unit_normal)
        return -tfd.Laplace(0.0, scale).log_prob(distances).sum()

    def initialize_state(pointcloud):
        return (jp.array([0.0, 1.0, 0.0]), jp.mean(pointcloud, axis=0))

    state = initialize_state(pointcloud) if state is None else state
    optimizer = optax.adam(0.05) if optimizer is None else optimizer
    loss = paz.lock(compute_laplace_loss, scale, pointcloud)
    (normal, centroid), losses = optimize(state, optimizer, loss, steps)
    offset = -jp.dot(normal, centroid)
    return normal, offset, centroid


def build_plane_to_world(world_up, normal, position):
    y_axis = normal / jp.linalg.norm(normal)
    x_axis = jp.cross(world_up, y_axis)
    x_axis = x_axis / jp.linalg.norm(x_axis)
    z_axis = jp.cross(y_axis, x_axis)
    rotation = jp.stack([x_axis, y_axis, z_axis], axis=1)
    return paz.SE3.to_affine_matrix(rotation, position)
