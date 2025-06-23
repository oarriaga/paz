import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import paz
import keras

camera = 4


def SSDCustom(path, score_thresh=0.30, IOU_thresh=0.45, top_k=200, draw=None):
    model = keras.saving.load_model(path)
    # model = keras.saving.load_model(
    #     path,
    #     custom_objects={
    #         "Conv2DNormalization": paz.models.layers.Conv2DNormalization
    #     },
    #     compile=True,
    # )

    # model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    # model.load_weights(path, by_name=True, skip_mismatch=True)

    # model.load_weights(path)
    boxes = paz.models.detection.single_shot_detector.build_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    label_colors = paz.draw.lincolor(len(names))
    if draw is None:
        draw = paz.partial(paz.draw.boxes2D, names=names, colors=label_colors)
    variances = [0.1, 0.1, 0.2, 0.2]
    apply_NMS = (len(names), IOU_thresh, top_k)
    apply_NMS = paz.lock(paz.detection.apply_per_class_NMS, *apply_NMS)
    return paz.applications.detectors.SSD(
        model, score_thresh, boxes, variances, apply_NMS, draw
    )


# pipeline = paz.applications.SSD300VOC()
# pipeline = SSDCustom("experiments/19-06-2025_12-27-06_SSD300/SSD300.weights.h5")
# pipeline = SSDCustom("experiments/19-06-2025_11-09-34_SSD300/SSD300.keras")
pipeline = SSDCustom("experiments/20-06-2025_17-13-52_SSD300/SSD300.keras")
# pipeline = paz.applications.SSD512COCO()
camera = paz.Camera(identifier=camera)
player = paz.VideoPlayer((480, 640), pipeline, camera)
player.run()
