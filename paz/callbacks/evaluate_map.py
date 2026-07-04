import gc

import keras
import paz


class EvaluateMAP(keras.callbacks.Callback):
    """Logs detector mean average precision every `period` epochs.

    The detector wraps the trained model, so it reflects the current weights.
    """

    def __init__(self, detector, images, ground_truths, num_classes, period,
                 difficulties=None, iou_thresh=0.5, use_07_metric=True):
        super().__init__()
        self.detector = detector
        self.images = images
        self.ground_truths = ground_truths
        self.num_classes = num_classes
        self.period = period
        self.difficulties = difficulties
        self.iou_thresh = iou_thresh
        self.use_07_metric = use_07_metric

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.period != 0:
            return
        args = self.detector, self.images, self.ground_truths, self.num_classes
        kwargs = {"difficulties": self.difficulties,
                  "iou_thresh": self.iou_thresh,
                  "use_07_metric": self.use_07_metric, "verbose": True}
        result = paz.evaluation.compute_mAP(*args, **kwargs)
        gc.collect()  # reclaim the per-eval detection arrays on low-RAM hosts
        if logs is not None:
            logs["mAP"] = float(result["mAP"])
        print("Epoch %d mAP: %.4f" % (epoch + 1, result["mAP"]))
