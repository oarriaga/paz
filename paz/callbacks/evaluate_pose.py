from keras.callbacks import Callback

from paz.backend.evaluation import compute_ADD, compute_ADI, is_correct_ADD


class EvaluatePose(Callback):
    def __init__(self, images, poses_true, points3D, diameter, predict_pose,
                 period=5, verbose=1):
        super().__init__()
        self.images = images
        self.poses_true = poses_true
        self.points3D = points3D
        self.diameter = diameter
        self.predict_pose = predict_pose
        self.period = period
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate on the first epoch too so CSVLogger captures the columns.
        if epoch != 0 and (epoch + 1) % self.period != 0:
            return
        logs = logs if logs is not None else {}
        ADD, ADI, correct, valid = 0.0, 0.0, 0, 0
        for image, pose_true in zip(self.images, self.poses_true):
            pose_pred = self.predict_pose(self.model, image)
            if pose_pred is None:
                continue
            add = compute_ADD(self.points3D, pose_true, pose_pred)
            ADD += add
            ADI += compute_ADI(self.points3D, pose_true, pose_pred)
            correct += int(is_correct_ADD(add, self.diameter))
            valid += 1
        if valid == 0:
            return
        logs["ADD"] = ADD / valid
        logs["ADI"] = ADI / valid
        logs["ADD_accuracy"] = correct / len(self.poses_true)
        if self.verbose:
            print(f" - ADD {logs['ADD']:.4f} - ADI {logs['ADI']:.4f} "
                  f"- ADD_accuracy {logs['ADD_accuracy']:.3f}")
