from collections import namedtuple

import jax
import jax.numpy as jp

from paz.backend.lie import SE3
from paz.optimization.robust import apply_huber
from paz.optimization.robust import huber_weights
from paz.slam.factors import compute_observation_jacobians
from paz.slam.factors import compute_observation_residuals

RESULT_FIELDS = ("poses", "landmarks", "initial_cost", "final_cost",
                 "initial_rmse", "final_rmse", "cost_trace",
                 "damping_trace", "num_accepted", "valid")
BundleAdjustment = namedtuple("BundleAdjustment", RESULT_FIELDS)


def bundle_adjust(problem, num_iterations, huber_scale, damping):
    def step(iteration, carry):
        problem, cost, damping, num_accepted = carry[:4]
        cost_trace, damping_trace = carry[4:]
        candidate = compute_candidate(problem, huber_scale, damping)
        candidate_cost = compute_cost(candidate, huber_scale)
        accept = is_acceptable(candidate, candidate_cost, cost)
        problem = select_candidate(accept, candidate, problem)
        cost = jp.where(accept, candidate_cost, cost)
        num_accepted = num_accepted + accept
        damping_trace = damping_trace.at[iteration].set(damping)
        damping = update_damping(damping, accept)
        cost_trace = cost_trace.at[iteration + 1].set(cost)
        return (problem, cost, damping, num_accepted, cost_trace,
                damping_trace)

    initial_cost = compute_cost(problem, huber_scale)
    initial_rmse = compute_rmse(problem)
    cost_trace = jp.zeros(num_iterations + 1).at[0].set(initial_cost)
    carry = (problem, initial_cost, jp.zeros(()) + damping)
    carry += (jp.zeros((), int), cost_trace, jp.zeros(num_iterations))
    carry = jax.lax.fori_loop(0, num_iterations, step, carry)
    problem, final_cost, _, num_accepted = carry[:4]
    cost_trace, damping_trace = carry[4:]
    final_rmse = compute_rmse(problem)
    valid = all_finite(problem, final_cost)
    results = (problem.poses, problem.landmarks, initial_cost, final_cost,
               initial_rmse, final_rmse, cost_trace, damping_trace,
               num_accepted, valid)
    return BundleAdjustment(*results)


def compute_candidate(problem, huber_scale, damping):
    args = (problem, huber_scale, damping)
    delta_poses, delta_landmarks = compute_schur_step(*args)
    retracted = jax.vmap(SE3.retract)(problem.poses, delta_poses)
    free = compute_free_pose_mask(problem)
    poses = jp.where(free[:, None, None], retracted, problem.poses)
    active = problem.landmark_active.astype(bool)
    moved = problem.landmarks + delta_landmarks
    landmarks = jp.where(active[:, None], moved, problem.landmarks)
    return problem._replace(poses=poses, landmarks=landmarks)


def compute_schur_step(problem, huber_scale, damping):
    residuals = compute_observation_residuals(problem)
    pose_blocks, landmark_blocks = compute_observation_jacobians(problem)
    weights = compute_irls_weights(problem, residuals, huber_scale)
    args = (problem, residuals, pose_blocks, landmark_blocks, weights)
    blocks = accumulate_normal_blocks(*args)
    pose_hessian, landmark_hessian, coupling = blocks[:3]
    pose_gradient, landmark_gradient = blocks[3:]
    pose_hessian = damp_blocks(pose_hessian, damping)
    landmark_hessian = damp_blocks(landmark_hessian, damping)
    free = compute_free_pose_mask(problem)
    gauge_args = (pose_hessian, coupling, pose_gradient, free)
    pose_hessian, coupling, pose_gradient = fix_gauge(*gauge_args)
    active = problem.landmark_active.astype(bool)
    landmark_hessian = mask_inactive_landmarks(landmark_hessian, active)
    solve_args = (pose_hessian, landmark_hessian, coupling)
    solve_args += (pose_gradient, landmark_gradient)
    return solve_schur(*solve_args)


def compute_irls_weights(problem, residuals, huber_scale):
    norms = jp.linalg.norm(residuals, axis=1)
    robust = huber_weights(norms, huber_scale)
    active = problem.observation_active.astype(residuals.dtype)
    return active * problem.observation_weight * robust


def accumulate_normal_blocks(problem, residuals, pose_blocks,
                             landmark_blocks, weights):
    num_poses, num_landmarks = len(problem.poses), len(problem.landmarks)
    pose_index = problem.observation_pose
    landmark_index = problem.observation_landmark
    scale, dtype = weights[:, None, None], residuals.dtype
    products = scale * outer_blocks(pose_blocks, pose_blocks)
    shape = (num_poses, 6, 6)
    pose_hessian = scatter_add(products, pose_index, shape, dtype)
    products = scale * outer_blocks(landmark_blocks, landmark_blocks)
    shape = (num_landmarks, 3, 3)
    landmark_hessian = scatter_add(products, landmark_index, shape, dtype)
    products = scale * outer_blocks(pose_blocks, landmark_blocks)
    coupling = jp.zeros((num_landmarks, num_poses, 6, 3), dtype)
    coupling = coupling.at[landmark_index, pose_index].add(products)
    gradients = weights[:, None] * pull_back(pose_blocks, residuals)
    shape = (num_poses, 6)
    pose_gradient = scatter_add(gradients, pose_index, shape, dtype)
    gradients = weights[:, None] * pull_back(landmark_blocks, residuals)
    shape = (num_landmarks, 3)
    landmark_gradient = scatter_add(gradients, landmark_index, shape, dtype)
    return (pose_hessian, landmark_hessian, coupling, pose_gradient,
            landmark_gradient)


def damp_blocks(blocks, damping):
    eye = jp.eye(blocks.shape[-1], dtype=blocks.dtype)
    return blocks + damping * eye * blocks + 1e-12 * eye


def compute_free_pose_mask(problem):
    indices = jp.arange(len(problem.poses))
    return problem.pose_active.astype(bool) & (indices != 0)


def fix_gauge(pose_hessian, coupling, pose_gradient, free):
    eye = jp.eye(6, dtype=pose_hessian.dtype)
    pose_hessian = jp.where(free[:, None, None], pose_hessian, eye)
    coupling = jp.where(free[None, :, None, None], coupling, 0.0)
    pose_gradient = jp.where(free[:, None], pose_gradient, 0.0)
    return pose_hessian, coupling, pose_gradient


def mask_inactive_landmarks(landmark_hessian, landmark_active):
    eye = jp.eye(3, dtype=landmark_hessian.dtype)
    return jp.where(landmark_active[:, None, None], landmark_hessian, eye)


def solve_schur(pose_hessian, landmark_hessian, coupling, pose_gradient,
                landmark_gradient):
    num_poses = pose_hessian.shape[0]
    landmark_inverse = jp.linalg.inv(landmark_hessian)
    gain = jp.einsum("lkab,lbc->lkac", coupling, landmark_inverse)
    gain_flat = flatten_coupling(gain)
    coupling_flat = flatten_coupling(coupling)
    reduced_hessian = build_block_diagonal(pose_hessian)
    reduced_hessian = reduced_hessian - gain_flat @ coupling_flat.T
    reduced_gradient = gain_flat @ landmark_gradient.reshape(-1)
    reduced_gradient = pose_gradient.reshape(-1) - reduced_gradient
    delta = jp.linalg.solve(reduced_hessian, -reduced_gradient)
    delta_poses = delta.reshape(num_poses, 6)
    coupled = jp.einsum("lkab,ka->lb", coupling, delta_poses)
    back_gradient = -(landmark_gradient + coupled)
    delta_landmarks = jp.einsum("lab,lb->la", landmark_inverse, back_gradient)
    return delta_poses, delta_landmarks


def flatten_coupling(coupling):
    num_landmarks, num_poses = coupling.shape[0], coupling.shape[1]
    stacked = coupling.transpose(1, 2, 0, 3)
    return stacked.reshape(num_poses * 6, num_landmarks * 3)


def build_block_diagonal(blocks):
    num_blocks, size = blocks.shape[0], blocks.shape[1]
    indices = jp.arange(num_blocks)
    dense = jp.zeros((num_blocks, size, num_blocks, size), blocks.dtype)
    dense = dense.at[indices, :, indices, :].set(blocks)
    return dense.reshape(num_blocks * size, num_blocks * size)


def outer_blocks(blocks_A, blocks_B):
    return jp.einsum("oar,oac->orc", blocks_A, blocks_B)


def pull_back(blocks, residuals):
    return jp.einsum("oab,oa->ob", blocks, residuals)


def scatter_add(values, indices, shape, dtype):
    return jp.zeros(shape, dtype).at[indices].add(values)


def compute_cost(problem, huber_scale):
    residuals = compute_observation_residuals(problem)
    squared_norms = jp.sum(residuals**2, axis=1)
    robust = apply_huber(squared_norms, huber_scale)
    active = problem.observation_active.astype(residuals.dtype)
    return jp.sum(active * problem.observation_weight * robust)


def compute_rmse(problem):
    residuals = compute_observation_residuals(problem)
    squared_norms = jp.sum(residuals**2, axis=1)
    active = problem.observation_active.astype(residuals.dtype)
    return jp.sqrt(jp.sum(active * squared_norms) / jp.sum(active))


def is_acceptable(candidate, candidate_cost, cost):
    finite_poses = jp.all(jp.isfinite(candidate.poses))
    finite_landmarks = jp.all(jp.isfinite(candidate.landmarks))
    finite = finite_poses & finite_landmarks & jp.isfinite(candidate_cost)
    return finite & (candidate_cost < cost)


def select_candidate(accept, candidate, problem):
    poses = jp.where(accept, candidate.poses, problem.poses)
    landmarks = jp.where(accept, candidate.landmarks, problem.landmarks)
    return problem._replace(poses=poses, landmarks=landmarks)


def all_finite(problem, cost):
    finite_poses = jp.all(jp.isfinite(problem.poses))
    finite_landmarks = jp.all(jp.isfinite(problem.landmarks))
    return finite_poses & finite_landmarks & jp.isfinite(cost)


def update_damping(damping, accept):
    scaled = jp.where(accept, damping / 3.0, damping * 3.0)
    return jp.clip(scaled, 1e-9, 1e6)
