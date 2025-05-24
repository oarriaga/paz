import jax.numpy as jp
import jax
import paz


def EFFICIENTDET(image, model, prior_boxes, score_thresh, IOU_thresh, top_k):
    mean = paz.image.RGB_IMAGENET_MEAN
    stdv = paz.image.RGB_IMAGENET_STDV
    image_size = paz.image.get_size(image)
    num_classes = model.output_shape[-1]
    model_size = model.input_shape[1:3]

    def preprocess(image):
        image = paz.cast(image, "float32")
        image = paz.image.subtract_mean(image, jp.array(mean))
        image = paz.image.divide_by_std(image, jp.array(stdv))
        image = paz.image.resize_pad_top_left(image, model_size, "linear")
        image = jp.expand_dims(image, axis=0)
        return image

    def postprocess(detections):
        detections = jp.squeeze(detections, axis=0)
        detections = paz.detection.decode(detections, prior_boxes, jp.ones(4))
        NMS_args = (num_classes, IOU_thresh, top_k, 0.01)
        detections = paz.detection.apply_per_class_NMS(detections, *NMS_args)
        detections = paz.detection.filter_by_score(detections, score_thresh, -1)
        return detections

    image = jax.jit(preprocess)(image)
    predictions = model(image)
    detections = jax.jit(postprocess, device=jax.devices("cpu")[0])(predictions)
    detections = paz.detection.remove_invalid(detections)
    detections = paz.detection.denormalize(detections, *image_size)
    boxes, class_args, scores = paz.detection.to_boxes2D(detections)
    return boxes, class_args, scores
