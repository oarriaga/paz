import jax.numpy as jp
import paz

MESH_TERMS = (
    "image",
    "depth",
    "masks",
    "smooth_mesh",
    "volume",
    "smooth_depth",
)


def scale_prior_loss(scale_vectors, scale_priors):
    priors = jp.array(scale_priors)[None, :]
    losses = paz.losses.mse(priors, scale_vectors, axis=-1, reduction="none")
    return losses.sum()


def material_loss(materials, curvature):
    barrier = paz.lock(paz.losses.soft_box_barrier, curvature)
    loss_1 = barrier(materials.color, 0.0, 1.0)
    loss_2 = barrier(materials.ambient, 0.0, 1.0)
    loss_3 = barrier(materials.diffuse, 0.0, 1.0)
    loss_4 = barrier(materials.specular, 0.0, 1.0)
    loss_5 = barrier(materials.shininess, 0.0, 200.0)
    return loss_1 + loss_2 + loss_3 + loss_4 + loss_5


def build_scene_loss(weights, materials, shapes, data):
    model, true_image, background_mask, curvature = data
    scene_loss = paz.lock(paz.losses.masked_mae, background_mask)

    def loss_fn(parameters):
        lights, floor_material = parameters
        pred_image, _, _, _ = model(lights, floor_material, materials, *shapes)
        color_loss = scene_loss(true_image, pred_image)
        barrier_loss = material_loss(materials, curvature)
        losses = [color_loss, barrier_loss]
        return paz.losses.weight(losses, weights)

    return loss_fn


def build_material_shape_loss(weights, scene_state, shapes, data):
    model, true_image, true_masks, curvature = data
    lights, floor_material = scene_state
    image_loss = paz.lock(paz.losses.masked_mae, true_masks)

    def loss_fn(materials):
        pred_image, _, _, _ = model(lights, floor_material, materials, *shapes)
        color_loss = image_loss(true_image, pred_image)
        barrier_loss = material_loss(materials, curvature)
        losses = [color_loss, barrier_loss]
        return paz.losses.weight(losses, weights)

    return loss_fn


def build_shape_loss(weights, scene_state, materials, data):
    model, true_depth, true_masks, scale_priors = data
    lights, floor_material = scene_state
    depth_loss = paz.lock(paz.losses.masked_mae, true_masks)

    def loss_fn(parameters):
        render_args = (lights, floor_material, materials)
        _, pred_depth, pred_masks, aux = model(*render_args, *parameters)
        scale_vectors_final = aux["final_scale_vectors"]
        losses = [
            depth_loss(true_depth, pred_depth),
            paz.losses.mae(true_masks, pred_masks),
            scale_prior_loss(scale_vectors_final, scale_priors),
        ]
        return paz.losses.weight(losses, weights)

    return loss_fn


def build_metrics(weights, scene_state, materials, data):
    model, true_depth, true_masks, scale_priors = data
    lights, floor_material = scene_state
    depth_loss = paz.lock(paz.losses.masked_mae, true_masks)

    def metrics_fn(parameters):
        render_args = (lights, floor_material, materials)
        _, pred_depth, pred_masks, aux = model(*render_args, *parameters)
        scale_vectors_final = aux["final_scale_vectors"]
        scale_loss = scale_prior_loss(scale_vectors_final, scale_priors)
        return {
            "depth": weights["depth"] * depth_loss(true_depth, pred_depth),
            "masks": weights["masks"] * paz.losses.mae(true_masks, pred_masks),
            "scale": weights["scale"] * scale_loss,
        }

    return metrics_fn


def build_mesh_loss(weights, model, observations, regularizer_data):
    terms = compute_mesh_terms(model, observations, regularizer_data)

    def loss_fn(cage_vertices):
        values = terms(cage_vertices)
        losses = [values[name] for name in MESH_TERMS]
        return paz.losses.weight(losses, weights)

    return loss_fn


def build_mesh_metrics(weights, model, observations, regularizer_data):
    terms = compute_mesh_terms(model, observations, regularizer_data)

    def metrics_fn(cage_vertices):
        values = terms(cage_vertices)
        return {name: weights[name] * values[name] for name in MESH_TERMS}

    return metrics_fn


def compute_mesh_terms(model, observations, regularizer_data):
    true_image, true_depth, true_masks = observations
    laplacian, initial_volumes, _ = regularizer_data
    target_volumes = initial_volumes

    def terms_fn(cage_vertices):
        pred_image, pred_depth, pred_masks, aux = model(cage_vertices)
        verts, faces = aux["vertices"], aux["faces"]
        smooth_mesh = paz.losses.laplacian_smoothing(
            verts, laplacian, reduction="sum")
        volume = paz.losses.volume_matching(
            verts, faces, target_volumes, reduction="mean")
        smooth_depth = paz.losses.depth.guided_smoothing(true_depth, pred_depth)
        return {
            "image": paz.losses.masked_mae(true_image, pred_image, true_masks),
            "depth": paz.losses.masked_mae(true_depth, pred_depth, true_masks),
            "masks": paz.losses.mae(true_masks, pred_masks),
            "smooth_mesh": smooth_mesh,
            "volume": volume,
            "smooth_depth": smooth_depth,
        }

    return terms_fn
