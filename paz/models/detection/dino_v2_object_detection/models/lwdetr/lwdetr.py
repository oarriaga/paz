from collections import namedtuple

import keras
from keras import Input, Model, layers, ops

from paz.models.detection.dino_v2_object_detection.utils import box_ops
from paz.models.detection.dino_v2_object_detection.utils.misc import interpolate
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (  # fmt: skip
    apply_segmentation_head,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (  # fmt: skip
    apply_transformer,
    apply_mlp,
    mlp,
)

QUERY_DIM = 4
FOCAL_GAMMA = 2
AUXILIARY_KEYS = ("aux_outputs", "enc_outputs")

CriterionArgs = namedtuple(
    "CriterionArgs",
    "num_classes matcher weight_dict loss_types focal_alpha group_detr "
    "sum_group_losses use_varifocal_loss use_position_supervised_loss "
    "ia_bce_loss mask_point_sample_ratio",
    defaults=(1, False, False, False, False, 16),
)


def expand_to_logits(loss, logits):
    if ops.ndim(loss) < ops.ndim(logits):
        loss = ops.expand_dims(loss, axis=-1)
    return loss


def binary_crossentropy_logits(inputs, targets):
    loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    return expand_to_logits(loss, inputs)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):  # fmt: skip
    probabilities = ops.sigmoid(inputs)
    entropy = binary_crossentropy_logits(inputs, targets)
    # p_t is the probability of the correct class, so (1 - p_t)^gamma
    # downweights well-classified examples.
    correct = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = entropy * ((1 - correct) ** gamma)
    if alpha >= 0:
        balance = alpha * targets + (1 - alpha) * (1 - targets)
        loss = balance * loss
    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha=0.75, gamma=2):  # fmt: skip
    probabilities = ops.sigmoid(inputs)
    # Positive samples weighted by target quality; negatives by focal
    # modulation of the prediction-target distance.
    positive = ops.cast(targets > 0.0, "float32")
    negative = ops.cast(targets <= 0.0, "float32")
    modulation = ops.abs(probabilities - targets) ** gamma
    weight = targets * positive + (1 - alpha) * modulation * negative
    loss = binary_crossentropy_logits(inputs, targets) * weight
    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):  # fmt: skip
    probabilities = ops.sigmoid(inputs)
    entropy = binary_crossentropy_logits(inputs, targets)
    loss = entropy * (ops.abs(targets - probabilities) ** gamma)
    if alpha >= 0:
        positive = ops.cast(targets > 0.0, "float32")
        negative = ops.cast(targets <= 0.0, "float32")
        loss = (alpha * positive + (1 - alpha) * negative) * loss
    return ops.sum(ops.mean(loss, axis=1)) / num_boxes


def dice_loss(inputs, targets, num_masks):
    inputs = ops.sigmoid(inputs)
    inputs = ops.reshape(inputs, (ops.shape(inputs)[0], -1))
    targets = ops.reshape(targets, (ops.shape(targets)[0], -1))
    numerator = 2 * ops.sum(inputs * targets, axis=-1)
    denominator = ops.sum(inputs, axis=-1) + ops.sum(targets, axis=-1)
    return ops.sum(1 - (numerator + 1) / (denominator + 1)) / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = ops.binary_crossentropy(targets, inputs, from_logits=True)
    return ops.sum(ops.mean(loss, axis=1)) / num_masks


@keras.saving.register_keras_serializable(package="lwdetr")
def query_indices(reference, num=0):
    # Registered so a ported .keras file using this arange Lambda can
    # deserialize the query/refpoint embedding index layer. `reference` is
    # the Lambda's required tensor input even though only `num` is read.
    return ops.arange(num, dtype="int32")


def build_query_indices(num, anchor, name):
    keys = ("arguments", "output_shape", "name")
    values = ({"num": num}, (num,), f"{name}_indices")
    return layers.Lambda(query_indices, **dict(zip(keys, values)))(anchor)


def lookup_table(num, dim, initializer, anchor, name):
    indices = build_query_indices(num, anchor, name)
    kwargs = dict(embeddings_initializer=initializer, name=name)
    return layers.Embedding(num, dim, **kwargs)(indices)


def LWDETR(backbone, transformer, segmentation_head, num_classes, num_queries, aux_loss=False, group_detr=1, two_stage=False, lite_refpoint_refine=False, bbox_reparam=False, name="lwdetr"):  # fmt: skip
    hidden_dim = transformer.d_model
    image, mask = build_lwdetr_inputs(backbone)
    features, _ = backbone([image, mask])
    anchor = read_anchor(features[0])
    tokens = ops.reshape(anchor, (ops.shape(anchor)[0], -1, hidden_dim))
    head_args = (num_classes, hidden_dim, QUERY_DIM, num_queries * group_detr)
    outputs = materialize_heads(tokens, anchor, *head_args, group_detr, two_stage)  # fmt: skip
    outputs = outputs + materialize_transformer(transformer, tokens)
    model = Model([image, mask], outputs, name=name)
    keys = ("backbone", "transformer", "segmentation_head", "num_classes", "num_queries", "hidden_dim", "aux_loss", "group_detr", "two_stage", "lite_refpoint_refine", "bbox_reparam")  # fmt: skip
    values = (backbone, transformer, segmentation_head, num_classes, num_queries, hidden_dim, aux_loss, group_detr, two_stage, lite_refpoint_refine, bbox_reparam)  # fmt: skip
    for key, value in zip(keys, values):
        setattr(model, key, value)
    return model


def build_lwdetr_inputs(backbone):
    image_input, mask_input = backbone.inputs[0], backbone.inputs[1]
    image = Input(batch_shape=image_input.shape, name="lwdetr_images")
    kwargs = dict(dtype=mask_input.dtype, name="lwdetr_mask")
    mask = Input(batch_shape=mask_input.shape, **kwargs)
    return image, mask


def read_anchor(feature):
    return feature[0] if isinstance(feature, (list, tuple)) else feature


def materialize_heads(tokens, anchor, num_classes, hidden_dim, query_dim, num_slots, group_detr, two_stage):  # fmt: skip
    zeros = keras.initializers.Zeros()
    glorot = keras.initializers.GlorotUniform()
    outputs = [layers.Dense(num_classes, name="class_embed")(tokens)]
    outputs.append(mlp(tokens, hidden_dim, hidden_dim, query_dim, 3, "bbox_embed"))  # fmt: skip
    outputs.append(lookup_table(num_slots, query_dim, zeros, anchor, "refpoint_embed"))  # fmt: skip
    outputs.append(lookup_table(num_slots, hidden_dim, glorot, anchor, "query_feat"))  # fmt: skip
    if two_stage:
        args = (tokens, num_classes, hidden_dim, query_dim, group_detr)
        outputs += materialize_encoder_heads(*args)
    return outputs


def materialize_encoder_heads(tokens, num_classes, hidden_dim, query_dim, group_detr):  # fmt: skip
    outputs = []
    for group in range(group_detr):
        class_name = f"enc_out_class_embed_{group}"
        bbox_name = f"enc_out_bbox_embed_{group}"
        outputs.append(layers.Dense(num_classes, name=class_name)(tokens))
        outputs.append(mlp(tokens, hidden_dim, hidden_dim, query_dim, 3, bbox_name))  # fmt: skip
    return outputs


def materialize_transformer(transformer, tokens):
    sine = ops.concatenate([tokens, tokens], axis=-1)
    return list(transformer([tokens, tokens, sine]))


def bbox_head(model, name):
    return lambda tokens: apply_mlp(model, tokens, 3, name)


def enc_class_heads(model):
    heads = None
    if model.two_stage:
        groups = range(model.group_detr)
        heads = [model.get_layer(f"enc_out_class_embed_{g}") for g in groups]
    return heads


def enc_bbox_heads(model):
    heads = None
    if model.two_stage:
        groups = range(model.group_detr)
        heads = [bbox_head(model, f"enc_out_bbox_embed_{g}") for g in groups]
    return heads


def set_aux_loss(outputs_class, outputs_coord, outputs_masks):
    aux_outputs = []
    for layer_index in range(ops.shape(outputs_class)[0] - 1):
        entry = {"pred_logits": outputs_class[layer_index]}
        entry["pred_boxes"] = outputs_coord[layer_index]
        if outputs_masks is not None:
            entry["pred_masks"] = outputs_masks[layer_index]
        aux_outputs.append(entry)
    return aux_outputs


def unpack_samples(samples):
    if isinstance(samples, (list, tuple)) and len(samples) == 2:
        tensors, mask = samples
    elif hasattr(samples, "tensors") and hasattr(samples, "mask"):
        tensors, mask = samples.tensors, samples.mask
    else:
        tensors, mask = samples, None
    tensors = ops.convert_to_tensor(tensors)
    if mask is None:
        # A zeros mask means "no padding" and matches the reference forward.
        mask = ops.zeros(ops.shape(tensors)[:3], dtype="bool")
    else:
        mask = ops.convert_to_tensor(mask)
    return tensors, mask


def split_backbone_features(features, mask):
    sources, masks = [], []
    for feature in features:
        source, feature_mask = split_feature(feature, mask)
        sources.append(source)
        masks.append(feature_mask)
    return sources, masks


def split_feature(feature, mask):
    if isinstance(feature, (list, tuple)):
        source, feature_mask = feature
    elif hasattr(feature, "decompose"):
        source, feature_mask = feature.decompose()
    else:
        source = feature
        feature_mask = resize_mask_to_feature(mask, source)
    return source, feature_mask


def resize_mask_to_feature(mask, source):
    if mask is None:
        resized = ops.cast(ops.zeros_like(source[..., 0]), "bool")
    else:
        size = ops.shape(source)[1:3]
        resized = interpolate(mask[:, None], size=size, mode="nearest")[:, 0]
    return resized


def select_query_tables(model, training):
    # Inference uses only the first query group; training uses all of them.
    refpoints = model.get_layer("refpoint_embed").embeddings
    queries = model.get_layer("query_feat").embeddings
    if not training:
        refpoints = refpoints[: model.num_queries]
        queries = queries[: model.num_queries]
    return queries, refpoints


def build_transformer_heads(model):
    decoder_bbox_embed = None
    if not model.lite_refpoint_refine:
        decoder_bbox_embed = bbox_head(model, "bbox_embed")
    return decoder_bbox_embed, enc_class_heads(model), enc_bbox_heads(model)


def read_image_size(tensors):
    if ops.ndim(tensors) == 4:
        size = ops.shape(tensors)[1:3]
    else:
        size = ops.shape(tensors)[0:2]
    return size


def read_spatial_features(model, sources):
    spatial = None
    if model.segmentation_head is not None:
        spatial = ops.transpose(sources[0], (0, 3, 1, 2))
    return spatial


def apply_lwdetr(model, samples, training=False):
    tensors, mask = unpack_samples(samples)
    backbone_input = [tensors, mask]
    features, positions = model.backbone(backbone_input, training=training)
    sources, masks = split_backbone_features(features, mask)
    heads = build_transformer_heads(model)
    queries = select_query_tables(model, training)
    args = (model.transformer, sources, masks, positions)
    hidden, references, hidden_enc, reference_enc = apply_transformer(*args, *heads, *queries, training)  # fmt: skip
    spatial = read_spatial_features(model, sources)
    image_size = read_image_size(tensors)
    outputs = build_detection_outputs(model, hidden, references, spatial, image_size)  # fmt: skip
    if model.two_stage:
        args = (model, outputs, hidden_enc, reference_enc, spatial)
        outputs = add_encoder_outputs(*args, image_size, training)
    return outputs


def build_detection_outputs(model, hidden, references, spatial, image_size):
    outputs = {}
    if hidden is not None:
        coordinates = decode_box_outputs(model, hidden, references)
        logits = model.get_layer("class_embed")(hidden)
        masks = build_mask_outputs(model, spatial, hidden, image_size)
        outputs = {"pred_logits": logits[-1], "pred_boxes": coordinates[-1]}
        if masks is not None:
            outputs["pred_masks"] = masks[-1]
        if model.aux_loss:
            outputs["aux_outputs"] = set_aux_loss(logits, coordinates, masks)
    return outputs


def decode_box_outputs(model, hidden, references):
    deltas = apply_mlp(model, hidden, 3, "bbox_embed")
    if model.bbox_reparam:
        # Deltas are relative to the reference points: the center is offset
        # by delta * reference_wh and the size scaled by exp(delta).
        centers = deltas[..., :2] * references[..., 2:] + references[..., :2]
        sizes = ops.exp(deltas[..., 2:]) * references[..., 2:]
        coordinates = ops.concatenate([centers, sizes], axis=-1)
    else:
        coordinates = ops.sigmoid(deltas + references)
    return coordinates


def build_mask_outputs(model, spatial, hidden, image_size):
    masks = None
    if model.segmentation_head is not None:
        args = (model.segmentation_head, spatial, hidden)
        masks = apply_segmentation_head(*args, image_size=image_size)
    return masks


def add_encoder_outputs(model, outputs, hidden_enc, reference_enc, spatial, image_size, training):  # fmt: skip
    group_detr = model.group_detr if training else 1
    grouped = ops.split(hidden_enc, group_detr, axis=1)
    heads = enc_class_heads(model)
    logits = [heads[group](grouped[group]) for group in range(group_detr)]
    encoded = {"pred_logits": ops.concatenate(logits, axis=1)}
    encoded["pred_boxes"] = reference_enc
    masks = build_encoder_mask_outputs(model, spatial, hidden_enc, image_size)
    if masks is not None:
        encoded["pred_masks"] = masks
    if outputs:
        outputs["enc_outputs"] = encoded
    else:
        outputs = encoded
    return outputs


def build_encoder_mask_outputs(model, spatial, hidden_enc, image_size):
    masks = None
    if model.segmentation_head is not None:
        args = (model.segmentation_head, spatial, [hidden_enc])
        kwargs = dict(image_size=image_size, skip_blocks=True)
        masks = apply_segmentation_head(*args, **kwargs)[0]
    return masks


def apply_lwdetr_stateless(model, trainable_variables, non_trainable_variables, samples, training=True):  # fmt: skip
    mapping = list(zip(model.trainable_variables, trainable_variables))
    mapping += zip(model.non_trainable_variables, non_trainable_variables)
    with keras.StatelessScope(state_mapping=mapping) as scope:
        outputs = apply_lwdetr(model, samples, training=training)
    return outputs, collect_updated_variables(model, scope)


def collect_updated_variables(model, scope):
    variables = model.non_trainable_variables
    return [scope.get_current_value(variable) for variable in variables]


def update_drop_path(model, drop_path_rate, vit_encoder_num_layers):
    encoder = model.backbone.get_layer("backbone").get_layer("encoder")
    num_layers = vit_encoder_num_layers or encoder.num_hidden_layers
    for depth_index in range(num_layers):
        rate = scale_drop_path_rate(drop_path_rate, depth_index, num_layers)
        for drop in find_drop_path_layers(encoder, depth_index):
            if isinstance(drop, layers.Dropout):
                drop.rate = rate


def scale_drop_path_rate(drop_path_rate, depth_index, num_layers):
    rate = 0.0
    if num_layers > 1:
        rate = drop_path_rate * depth_index / max(1, num_layers - 1)
    return rate


def find_drop_path_layers(encoder, depth_index):
    prefix = f"encoder_layer_{depth_index}_drop_path"
    try:
        found = [encoder.get_layer(f"{prefix}{slot}") for slot in (1, 2)]
    except ValueError:
        found = []
    return found


def update_dropout(model, dropout_rate):
    for layer in model._flatten_layers():
        if isinstance(layer, layers.Dropout):
            layer.rate = dropout_rate


def get_src_permutation_idx(indices):
    batch = [ops.full_like(source, index)
             for index, (source, _) in enumerate(indices)]
    sources = [source for (source, _) in indices]
    return ops.concatenate(batch), ops.concatenate(sources)


def gather_matched_labels(targets, indices):
    matched = [ops.take(target["labels"], columns, axis=0)
               for target, (_, columns) in zip(targets, indices)]
    return ops.concatenate(matched)


def gather_matched_boxes(targets, indices):
    matched = [ops.take(target["boxes"], columns, axis=0)
               for target, (_, columns) in zip(targets, indices)]
    return ops.concatenate(matched, axis=0)


def take_flat_rows(tensor, flat_index):
    columns = ops.shape(tensor)[-1]
    return ops.take(ops.reshape(tensor, (-1, columns)), flat_index, axis=0)


def compute_matched_iou(source_boxes, target_boxes):
    source_xyxy = box_ops.box_cxcywh_to_xyxy(ops.stop_gradient(source_boxes))
    target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
    iou_matrix, _ = box_ops.box_iou(source_xyxy, target_xyxy)
    return ops.stop_gradient(ops.diag(iou_matrix))


def build_ia_bce_weights(source_logits, flat_index, quality, alpha):
    probabilities = ops.sigmoid(source_logits)
    matched = ops.take(ops.reshape(probabilities, (-1,)), flat_index)
    # Soft target: probability^alpha * iou^(1-alpha), clamped at 0.01.
    soft = ops.maximum(matched**alpha * quality ** (1 - alpha), 0.01)
    soft = ops.stop_gradient(soft)
    scatter_index = ops.expand_dims(flat_index, axis=-1)
    positive = ops.reshape(ops.zeros_like(source_logits), (-1,))
    negative = ops.reshape(probabilities**FOCAL_GAMMA, (-1,))
    positive = ops.scatter_update(positive, scatter_index, ops.cast(soft, positive.dtype))  # fmt: skip
    negative = ops.scatter_update(negative, scatter_index, 1.0 - ops.cast(soft, negative.dtype))  # fmt: skip
    shape = ops.shape(source_logits)
    return ops.reshape(positive, shape), ops.reshape(negative, shape)


def reduce_ia_bce_loss(source_logits, weights, num_boxes):
    positive, negative = weights
    # Numerically stable form of the weighted BCE:
    # negative * logits - log_sigmoid(logits) * (positive + negative).
    log_sigmoid = -ops.softplus(-source_logits)
    loss = negative * source_logits - log_sigmoid * (positive + negative)
    return ops.sum(loss) / num_boxes


def compute_ia_bce_loss(outputs, targets, indices, index, target_classes, num_boxes, args):  # fmt: skip
    source_logits = outputs["pred_logits"]
    num_queries = ops.shape(source_logits)[1]
    num_classes = ops.shape(source_logits)[2]
    source_boxes = take_flat_rows(outputs["pred_boxes"], index[0] * num_queries + index[1])  # fmt: skip
    quality = compute_matched_iou(source_boxes, gather_matched_boxes(targets, indices))  # fmt: skip
    offsets = index[0] * num_queries * num_classes + index[1] * num_classes
    flat_index = offsets + ops.cast(target_classes, index[0].dtype)
    weights = build_ia_bce_weights(source_logits, flat_index, quality, args.focal_alpha)  # fmt: skip
    return reduce_ia_bce_loss(source_logits, weights, num_boxes)


def compute_focal_label_loss(source_logits, index, target_classes, num_boxes, args):  # fmt: skip
    filled = ops.full(source_logits.shape[:2], args.num_classes, dtype="int64")
    scattered = ops.scatter_update(filled, ops.stack(index, axis=-1), target_classes)  # fmt: skip
    num_classes = ops.shape(source_logits)[2]
    one_hot = ops.one_hot(scattered, num_classes + 1)[..., :-1]
    kwargs = dict(alpha=args.focal_alpha, gamma=FOCAL_GAMMA)
    loss = sigmoid_focal_loss(source_logits, one_hot, num_boxes, **kwargs)
    return loss * ops.cast(ops.shape(source_logits)[1], "float32")


# The four criterion_loss_* functions share one dispatch signature so that
# get_loss can invoke any of them by name; the parameters an individual loss
# does not read are required by that uniform contract.
def criterion_loss_labels(outputs, targets, indices, num_boxes, args, log=True):
    source_logits = outputs["pred_logits"]
    index = get_src_permutation_idx(indices)
    target_classes = gather_matched_labels(targets, indices)
    if args.ia_bce_loss:
        args_bce = (outputs, targets, indices, index, target_classes)
        loss = compute_ia_bce_loss(*args_bce, num_boxes, args)
    else:
        args_focal = (source_logits, index, target_classes, num_boxes, args)
        loss = compute_focal_label_loss(*args_focal)
    return {"loss_ce": loss}


def criterion_loss_boxes(outputs, targets, indices, num_boxes, args):
    rows, columns = get_src_permutation_idx(indices)
    num_queries = ops.shape(outputs["pred_boxes"])[1]
    flat_index = rows * num_queries + columns
    source_boxes = take_flat_rows(outputs["pred_boxes"], flat_index)
    target_boxes = gather_matched_boxes(targets, indices)
    loss_bbox = ops.sum(ops.abs(source_boxes - target_boxes)) / num_boxes
    source_xyxy = box_ops.box_cxcywh_to_xyxy(source_boxes)
    target_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
    giou = ops.diag(box_ops.generalized_box_iou(source_xyxy, target_xyxy))
    return {"loss_bbox": loss_bbox, "loss_giou": ops.sum(1 - giou) / num_boxes}


def criterion_loss_cardinality(outputs, targets, indices, num_boxes, args, **kwargs):  # fmt: skip
    logits = outputs["pred_logits"]
    lengths = [len(target["labels"]) for target in targets]
    lengths = ops.convert_to_tensor(lengths, dtype="float32")
    # Count predictions whose argmax is not the last (background) class.
    background = ops.shape(logits)[-1] - 1
    predicted = ops.cast(ops.argmax(logits, axis=-1) != background, "int32")
    counts = ops.cast(ops.sum(predicted, axis=1), "float32")
    error = ops.mean(ops.abs(counts - lengths))
    return {"cardinality_error": ops.stop_gradient(error)}


def gather_matched_masks(pred_masks, indices):
    batch = ops.shape(pred_masks)[0]
    num_queries = ops.shape(pred_masks)[1]
    height, width = ops.shape(pred_masks)[2], ops.shape(pred_masks)[3]
    flat = ops.reshape(pred_masks, (batch * num_queries, height, width))
    rows, columns = get_src_permutation_idx(indices)
    return ops.take(flat, rows * num_queries + columns, axis=0)


def gather_target_masks(targets, indices):
    matched = [ops.take(target["masks"], columns, axis=0)
               for target, (_, columns) in zip(targets, indices)
               if "masks" in target]
    return ops.concatenate(matched, axis=0) if matched else None


def match_mask_pairs(outputs, targets, indices):
    matched = None
    if "pred_masks" in outputs:
        source_masks = gather_matched_masks(outputs["pred_masks"], indices)
        target_masks = gather_target_masks(targets, indices)
        if target_masks is not None and ops.shape(source_masks)[0] != 0:
            matched = source_masks, target_masks
    return matched


def resize_mask_stack(masks, size, interpolation):
    expanded = ops.expand_dims(masks, axis=-1)
    return ops.image.resize(expanded, size, interpolation=interpolation)[..., 0]


def resize_target_masks(target_masks, height, width):
    mismatched = ops.shape(target_masks)[1] != height
    mismatched = mismatched or ops.shape(target_masks)[2] != width
    if mismatched:
        target_masks = ops.cast(target_masks, "float32")
        target_masks = resize_mask_stack(target_masks, (int(height), int(width)), "nearest")  # fmt: skip
    return target_masks


def downsample_mask_pair(source_masks, target_masks, height, width, ratio):
    # Downsample by mask_point_sample_ratio to keep the loss affordable.
    target_masks = ops.cast(target_masks, "float32")
    if ratio > 1:
        size = (max(1, int(height) // ratio), max(1, int(width) // ratio))
        source_masks = resize_mask_stack(source_masks, size, "bilinear")
        target_masks = resize_mask_stack(target_masks, size, "nearest")
    return source_masks, target_masks


def compute_mask_losses(source_masks, target_masks, num_boxes, args):
    height, width = ops.shape(source_masks)[1], ops.shape(source_masks)[2]
    target_masks = resize_target_masks(target_masks, height, width)
    args_downsample = (source_masks, target_masks, height, width)
    ratio = args.mask_point_sample_ratio
    source_masks, target_masks = downsample_mask_pair(*args_downsample, ratio)
    source_flat = ops.reshape(source_masks, (ops.shape(source_masks)[0], -1))
    target_flat = ops.reshape(target_masks, (ops.shape(target_masks)[0], -1))
    mask_ce = sigmoid_ce_loss(source_flat, target_flat, num_boxes)
    mask_dice = dice_loss(source_flat, target_flat, num_boxes)
    return {"loss_mask_ce": mask_ce, "loss_mask_dice": mask_dice}


def criterion_loss_masks(outputs, targets, indices, num_boxes, args, **kwargs):
    zero = ops.convert_to_tensor(0.0, dtype="float32")
    result = {"loss_mask_ce": zero, "loss_mask_dice": zero}
    matched = match_mask_pairs(outputs, targets, indices)
    if matched is not None:
        result = compute_mask_losses(*matched, num_boxes, args)
    return result


LOSS_FUNCTIONS = {
    "labels": criterion_loss_labels,
    "boxes": criterion_loss_boxes,
    "cardinality": criterion_loss_cardinality,
    "masks": criterion_loss_masks,
}


def get_loss(loss, outputs, targets, indices, num_boxes, args, **kwargs):
    losses = {}
    if loss in LOSS_FUNCTIONS:
        compute = LOSS_FUNCTIONS[loss]
        losses = compute(outputs, targets, indices, num_boxes, args, **kwargs)
    return losses


def normalize_box_count(targets, args, group_detr):
    num_boxes = sum(len(target["labels"]) for target in targets)
    if not args.sum_group_losses:
        num_boxes = num_boxes * group_detr
    return ops.cast(ops.maximum(num_boxes, 1), "float32")


def compute_loss_group(outputs, targets, indices, num_boxes, args, suffix, **kwargs):  # fmt: skip
    losses = {}
    for loss in args.loss_types:
        args_loss = (loss, outputs, targets, indices, num_boxes, args)
        computed = get_loss(*args_loss, **kwargs)
        losses.update({key + suffix: value for key, value in computed.items()})
    return losses


def compute_aux_losses(outputs, targets, num_boxes, args, group_detr):
    losses = {}
    if "aux_outputs" in outputs:
        for index, aux in enumerate(outputs["aux_outputs"]):
            indices = args.matcher(aux, targets, group_detr=group_detr)
            args_group = (aux, targets, indices, num_boxes, args)
            losses.update(compute_loss_group(*args_group, f"_{index}"))
    return losses


def compute_encoder_losses(outputs, targets, num_boxes, args, group_detr):
    losses = {}
    if "enc_outputs" in outputs:
        encoded = outputs["enc_outputs"]
        indices = args.matcher(encoded, targets, group_detr=group_detr)
        for loss in args.loss_types:
            kwargs = {"log": False} if loss == "labels" else {}
            args_group = (encoded, targets, indices, num_boxes, args)
            losses.update(compute_loss_group(*args_group, "_enc", **kwargs))
    return losses


def set_criterion(outputs, targets, args, training=True):
    group_detr = args.group_detr if training else 1
    main = {k: v for k, v in outputs.items() if k not in AUXILIARY_KEYS}
    indices = args.matcher(main, targets, group_detr=group_detr)
    num_boxes = normalize_box_count(targets, args, group_detr)
    args_group = (outputs, targets, indices, num_boxes, args)
    losses = compute_loss_group(*args_group, "")
    args_extra = (outputs, targets, num_boxes, args, group_detr)
    losses.update(compute_aux_losses(*args_extra))
    losses.update(compute_encoder_losses(*args_extra))
    return losses


def flatten_scores(logits):
    probabilities = ops.sigmoid(logits)
    return ops.reshape(probabilities, (ops.shape(logits)[0], -1))


def select_scaled_boxes(boxes, query_index, target_sizes, num_select):
    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    corners = box_ops.box_cxcywh_to_xyxy(boxes)
    batch, num_queries, coordinates = ops.shape(corners)
    flat = ops.reshape(corners, (-1, coordinates))
    offsets = ops.arange(batch)[:, None] * num_queries + query_index
    selected = ops.take(flat, ops.reshape(offsets, (-1,)), axis=0)
    selected = ops.reshape(selected, (batch, num_select, coordinates))
    height, width = target_sizes[:, 0], target_sizes[:, 1]
    scale = ops.stack([width, height, width, height], axis=1)
    return selected * ops.cast(scale, "float32")[:, None, :]


def resize_masks_to_target_sizes(masks, query_index, target_sizes):
    # Per-image mask resize is host-side by design: the output size differs
    # per image. One host sync for all sizes, not two per image.
    sizes = ops.convert_to_numpy(target_sizes)
    resized = []
    for index in range(sizes.shape[0]):
        selected = ops.take(masks[index], query_index[index], axis=0)
        size = (int(sizes[index, 0]), int(sizes[index, 1]))
        expanded = ops.expand_dims(selected, axis=-1)
        image = ops.image.resize(expanded, size, interpolation="bilinear")
        resized.append(ops.transpose(image, (0, 3, 1, 2)) > 0.0)
    return resized


def post_process(outputs, target_sizes, num_select=300):
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    scores, ranked = ops.top_k(flatten_scores(logits), num_select)
    num_classes = ops.shape(logits)[2]
    query_index = ranked // num_classes
    labels = ranked % num_classes
    boxes = select_scaled_boxes(boxes, query_index, target_sizes, num_select)
    masks = outputs.get("pred_masks", None)
    if masks is None:
        result = scores, labels, boxes
    else:
        args = (masks, query_index, target_sizes)
        result = scores, labels, boxes, resize_masks_to_target_sizes(*args)
    return result
