import jax.numpy as jp
import paz
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def build_observation_model(render, floor):
    """Builds observation model for Bayesian inverse graphics.

    Args:
        render: JIT-compiled paz.graphics.render function (partially applied with shadows)
        floor: paz.graphics.Shape representing the floor plane

    Returns:
        Function that converts samples to rendered images
    """
    zero_image = jp.zeros_like(floor.pattern.image)
    zero_pattern = paz.graphics.Pattern(jp.eye(4), paz.graphics.NO_PATTERN, zero_image)

    shape_types = jp.array(
        [paz.graphics.SPHERE, paz.graphics.CUBE, paz.graphics.CYLINDER]
    )

    def sample_to_shape(sample):
        shift = jp.squeeze(sample["shift"])
        scale = jp.squeeze(sample["scale"])
        color = jp.squeeze(sample["color"])
        ambient = jp.squeeze(sample["ambient"])
        diffuse = jp.squeeze(sample["diffuse"])
        specular = jp.squeeze(sample["specular"])
        shininess = jp.squeeze(sample["shininess"])
        classes = jp.squeeze(sample["classes"])
        theta = jp.squeeze(sample["theta"])

        x = shift[..., 0]
        z = shift[..., 1]
        y = scale[..., 1]
        translate = jp.array([x, y, z])
        translate = paz.SE3.translation(translate)
        rotate = paz.SE3.rotation_y(theta)
        scale = paz.SE3.scaling(scale)
        transform = translate @ rotate @ scale

        material = paz.graphics.Material(
            color=color,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shininess=shininess,
        )

        shape_type_arg = jp.argmax(classes)
        shape_type = shape_types[shape_type_arg]

        return paz.graphics.Shape(transform, shape_type, material, zero_pattern)

    def apply(sample):
        shape = sample_to_shape(sample)
        scene = paz.graphics.Scene([floor, shape])
        image, depth = render(scene=scene)
        image = jp.clip(image, 0.0, 1.0)
        return image, depth

    return apply


def build_render_function(image_shape, y_FOV, camera_pose, lights, shadows=False):
    """Builds a JIT-compiled render function for the observation model.

    Args:
        image_shape: Tuple (H, W) for image dimensions
        y_FOV: Float, vertical field of view in radians
        camera_pose: Array (4, 4), camera SE3 transformation
        lights: paz.graphics.PointLight or list of lights
        shadows: Boolean, whether to use shadows (static argument)

    Returns:
        JIT-compiled render function
    """
    H, W = image_shape
    rays = paz.graphics.camera.build_rays((H, W), y_FOV, camera_pose)
    render_args = ((H, W), camera_pose, rays)
    render_kwargs = {"lights": lights, "mask": None, "shadows": shadows}
    import jax
    from functools import partial

    return jax.jit(partial(paz.graphics.render, *render_args, **render_kwargs))


def denormalize_image(image):
    return 255.0 * image


def preprocess_input(image, mean=[103.939, 116.779, 123.68]):
    image = image.astype(jp.float32)
    image = image[..., ::-1]
    image = jp.moveaxis(image, 2, 0)
    mean = jp.array(mean)
    mean = jp.expand_dims(mean, axis=[1, 2])
    return image - mean


def compute_feature_loss(true_features, pred_features):
    losses = []
    for true_feature, pred_feature in zip(true_features, pred_features):
        loss = (true_feature - pred_feature) ** 2
        loss = jp.mean(loss, axis=(0, 1))
        losses.append(loss)
    return jp.array(losses)


def build_neuro_likelihood(weight, branch_model):
    """Build neural likelihood function using perceptual features.

    Args:
        weight: Float, weight for perceptual loss
        branch_model: Neural network for feature extraction

    Returns:
        Function computing perceptual log likelihood
    """

    def apply(true_image, pred_image):
        true_image = denormalize_image(true_image)
        pred_image = denormalize_image(pred_image)
        true_image = preprocess_input(true_image)
        pred_image = preprocess_input(pred_image)
        true_features = branch_model(true_image)
        pred_features = branch_model(pred_image)
        losses = compute_feature_loss(true_features, pred_features)
        return -weight * (losses.sum())

    return apply


def build_likelihood(observation_model, noise_model, neuro_model=None):
    """Build likelihood function for Bayesian inference.

    Args:
        observation_model: Function mapping samples to images
        noise_model: TFP distribution for pixel noise
        neuro_model: Optional perceptual likelihood function

    Returns:
        Function computing total log likelihood
    """

    def apply(forward_samples, true_image):
        pred_image, pred_depth = observation_model(forward_samples)
        color_log_prob = noise_model.log_prob(true_image - pred_image).sum()
        if neuro_model is not None:
            neuro_log_prob = neuro_model(true_image, pred_image)
            return color_log_prob + neuro_log_prob
        return color_log_prob

    return apply


def parse_summary(summary, statistic="mean"):
    """Parse MCMC summary to extract point estimates.

    Args:
        summary: ArviZ summary object
        statistic: String, which statistic to extract

    Returns:
        Dictionary of parameter values
    """
    def read_stat(name):
        row = summary[statistic]
        if name in row:
            return row[name]
        indexed = f"{name}[0]"
        if indexed in row:
            return row[indexed]
        raise KeyError(name)

    x_shift = summary[statistic]["shift[0]"]
    y_shift = summary[statistic]["shift[1]"]
    shift = jp.array([x_shift, y_shift])

    theta = jp.array(summary["mode"]["theta"])

    x_scale = summary[statistic]["scale[0]"]
    y_scale = summary[statistic]["scale[1]"]
    z_scale = summary[statistic]["scale[2]"]
    scale = jp.array([x_scale, y_scale, z_scale])

    r_color = summary[statistic]["color[0]"]
    g_color = summary[statistic]["color[1]"]
    b_color = summary[statistic]["color[2]"]
    color = jp.array([r_color, g_color, b_color])

    ambient = jp.array(read_stat("ambient"))
    diffuse = jp.array(read_stat("diffuse"))
    specular = jp.array(read_stat("specular"))
    shininess = jp.array(read_stat("shininess"))

    class_0 = summary[statistic]["classes[0]"]
    class_1 = summary[statistic]["classes[1]"]
    class_2 = summary[statistic]["classes[2]"]
    classes = jp.array([class_0, class_1, class_2])

    return {
        "shift": shift,
        "theta": theta,
        "scale": scale,
        "color": color,
        "ambient": ambient,
        "diffuse": diffuse,
        "specular": specular,
        "shininess": shininess,
        "classes": classes,
    }


def estimate_point(summary, render, statistic="mean"):
    """Render image from point estimate in MCMC summary.

    Args:
        summary: ArviZ summary object
        render: Observation model function
        statistic: String, which statistic to extract

    Returns:
        Tuple of (image, depth) arrays
    """
    sample = parse_summary(summary, statistic)
    point_image, point_depth = render(sample)
    return point_image, point_depth
