"""Predicting 3d poses from 2d joints"""
import os
import sys
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.callbacks import CallbackList
from absl import app, flags, logging
from absl.flags import FLAGS
from datetime import datetime
import cameras
import data_utils
from linear_model import LinearModel
from linear_model import mse_loss, get_all_batches
import procrustes

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")
flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
flags.DEFINE_boolean("max_norm", False, "Apply maxnorm constraint to the weights")
flags.DEFINE_boolean("batch_norm", False, "Use batch_normalization")

# Data loading
flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
flags.DEFINE_string("action", "All", "The action to train on. 'All' means all the actions")

# Architecture
flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
flags.DEFINE_boolean("residual", False, "Whether to add a residual connection every 2 layers")

# Evaluation
flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")
flags.DEFINE_boolean("evaluateActionWise", False, "The dataset to use either h36m or heva")

# Directories
flags.DEFINE_string("cameras_path", "data/h36m/metadata.xml", "File with h36m metadata, including cameras")
flags.DEFINE_string("data_dir", "SCRATCH/3d-pose-baseline/data/h36m/", "Data directory")
flags.DEFINE_string("train_dir", "SCRATCH/3d-pose-baseline/experiments", "Training directory.")

# Train or load
flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

# Misc
flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")


class CustomCallbackList(CallbackList):
    """This Class avoids the warning printed when callback takes more time as compared to training"""

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)

        logs = logs or {}
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)


tf.keras.callbacks.CallbackList = CustomCallbackList  # tf.python.keras.callbacks.CallbackList


def get_train_dir():
    return os.path.join(FLAGS.train_dir,
                        FLAGS.action,
                        'dropout_{0}'.format(FLAGS.dropout),
                        'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
                        'lr_{0}'.format(FLAGS.learning_rate),
                        'residual' if FLAGS.residual else 'not_residual',
                        'depth_{0}'.format(FLAGS.num_layers),
                        'linear_size{0}'.format(FLAGS.linear_size),
                        'batch_size_{0}'.format(FLAGS.batch_size),
                        'procrustes' if FLAGS.procrustes else 'no_procrustes',
                        'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
                        'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
                        'predict_14' if FLAGS.predict_14 else 'predict_17')


def denormalize(enc_in, dec_out, poses3d, data_mean_2d, data_std_2d,
                dim_to_ignore_2d, data_mean_3d, data_std_3d, dim_to_ignore_3d):
    """
    Function that denormalizes the inputs

    Args
      data_mean_3d: the mean of the training data in 3d
      data_std_3d: the standard deviation of the training data in 3d
      dim_to_use_3d: out of all the 96 dimensions that represent a 3d body in h36m, compute results for this subset
      dim_to_ignore_3d: complelment of the above
      data_mean_2d: mean of the training data in 2d
      data_std_2d: standard deviation of the training data in 2d
      dim_to_use_2d: out of the 64 dimensions that represent a body in 2d in h35m, use this subset
      dim_to_ignore_2d: complement of the above
      encoder_inputs: input for the network
      decoder_outputs: expected output for the network

    Returns
      enc_in: denormalized encoder inputs
      dec_out: adenormalized decoder outputs
      poses3d: denormalized 3D poses
    """
    # denormalize
    enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
    dec_out = data_utils.unNormalizeData(dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d)
    poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)

    return enc_in, dec_out, poses3d


def calculate_err(poses3d, dec_out, n_joints, all_dists):
    """
    Compute Euclidean distance error per joint

    Args
      poses3d: predicted 3d poses
      dec_out: expected output for the network
      n_joints: number of joints
      all_dists: empty list
    Returns
      all_dists: list of L2 distance
    """
    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out) ** 2  # Squared error between prediction and expected output
    dists = np.zeros((sqerr.shape[0], n_joints))  # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints * 3, 3):
        # Sum across X,Y, and Z dimenstions to obtain L2 distance
        dists[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
        dist_idx = dist_idx + 1

    all_dists.append(dists)

    return all_dists


def lr_exp_decay(epoch):
    n_batch = 24371
    decay_rate = 0.96
    decay_steps = 100000
    p = ((epoch + 1) * n_batch) / decay_steps
    lr = tf.multiply(FLAGS.learning_rate, tf.pow(decay_rate, p))
    tf.summary.scalar('learning_rate', data=lr, step=epoch)
    return lr


class ValCallback(Callback):
    def on_test_batch_end(self, batch, logs=None):
        if (batch + 1) % 1000 == 603:
            print("...Evaluating: batch {}".format(batch + 1))


class EvaluateEpoch(Callback):
    def __init__(self, enc_inputs_val, dec_outputs_val, model, data_mean_2d, data_std_2d, dim_to_ignore_2d,
                 data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d, logdir):
        super(EvaluateEpoch, self).__init__()
        self.enc_in = enc_inputs_val
        self.dec_out = dec_outputs_val
        self.model = model
        self.dm_2d = data_mean_2d
        self.ds_2d = data_std_2d
        self.dti_2d = dim_to_ignore_2d
        self.dm_3d = data_mean_3d
        self.ds_3d = data_std_3d
        self.dtu_3d = dim_to_use_3d
        self.dti_3d = dim_to_ignore_3d
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs=None):
        n_joints = 17
        all_dists, start_time, loss = [], time.time(), 0.

        print("==> Working on test epoch {0}".format(epoch + 1))

        dp = 1.0  # dropout keep probability is always 1 at test time
        poses3d = self.model.predict(x=self.enc_in)
        poses3d = np.asarray(poses3d)
        step_loss = mse_loss(self.dec_out, poses3d)
        loss += step_loss

        # denormalize
        enc_in, dec_out, poses3d = denormalize(self.enc_in, self.dec_out, poses3d, self.dm_2d, self.ds_2d,
                                               self.dti_2d, self.dm_3d, self.ds_3d, self.dti_3d)

        # Keep only the relevant dimensions
        dtu3d = np.hstack((np.arange(3), self.dtu_3d))
        dec_out = dec_out[:, dtu3d]
        poses3d = poses3d[:, dtu3d]

        if FLAGS.procrustes:
            # Apply per-frame procrustes alignment if asked to do so
            for j in range(FLAGS.batch_size):
                gt = np.reshape(dec_out[j, :], [-1, 3])
                out = np.reshape(poses3d[j, :], [-1, 3])
                _, Z, T, b, c = procrustes.compute_similarity_transform(gt, out, compute_optimal_scale=True)
                out = (b * out.dot(T)) + c

                poses3d[j, :] = np.reshape(out, [-1, 17 * 3])

        all_dists = calculate_err(poses3d, dec_out, n_joints, all_dists)
        step_time = (time.time() - start_time)
        all_dists = np.vstack(all_dists)

        # Error per joint and total for all passed batches
        joint_err = np.mean(all_dists, axis=0)
        total_err = np.mean(all_dists)

        print("=============================\n"
              "Step-time (s):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % (step_time, loss, total_err))

        for i in range(n_joints):
            # 6 spaces, right-aligned, 5 decimal places
            print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i + 1, joint_err[i]))
        print("=============================")

        # Saving the evaluation results
        filename = os.path.join(self.logdir, 'Val_log.txt')
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'a') as eval_log_file:
            eval_log_file.write('Epoch: {}, total_err: {}, joint_err: {}, loss: {}, step_time: {} s\n'.
                                format(str(epoch), total_err, joint_err, loss, step_time))


def train():
    """Train a linear model for 3d pose estimation"""
    train_dir = get_train_dir()
    print(f"==> train_dir {train_dir}")

    # Logs dir for train and test runs
    path_prefix = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(path_prefix, 'logs/scalars/', datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_dir = os.path.join(path_prefix, 'saved_model/')

    if not os.path.exists(save_dir):
        os.system('mkdir -p {}'.format(save_dir))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    actions = data_utils.define_actions(FLAGS.action)

    # Load camera parameters
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    this_file = os.path.dirname(os.path.realpath(__file__))
    rcams = cameras.load_cameras(os.path.join(this_file, "..", FLAGS.cameras_path), SUBJECT_IDS)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, \
    test_root_positions = data_utils.read_3d_data(actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    # Read groundtruth 2D projections
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data(
        actions, FLAGS.data_dir, rcams)
    print("\n==> done reading and normalizing data.")

    # === Create the model ===
    print("\n==> Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = LinearModel(
        FLAGS.linear_size,
        FLAGS.num_layers,
        FLAGS.residual,
        FLAGS.max_norm,
        FLAGS.batch_norm,
        FLAGS.dropout,
        FLAGS.predict_14,
    )
    print("\n==> Model created!")

    # Define loss function (criterion) and optimizer and compile model

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
    )
    optimizer = Adam(learning_rate=FLAGS.learning_rate)  # lr_schedule
    criterion = mse_loss
    model.compile(optimizer, criterion)  # model.compile(opt, loss) is syntax

    encoder_inputs, decoder_outputs = get_all_batches(train_set_2d, train_set_3d, FLAGS.camera_frame,
                                                      FLAGS.batch_size)

    enc_inputs_val, dec_outputs_val = get_all_batches(test_set_2d, test_set_3d, FLAGS.camera_frame,
                                                      FLAGS.batch_size)

    # Callbacks
    callbacks = [LearningRateScheduler(lr_exp_decay, verbose=1),
                 EvaluateEpoch(enc_inputs_val, dec_outputs_val, model, data_mean_2d, data_std_2d, dim_to_ignore_2d,
                               data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d, logdir),
                 ValCallback(),
                 TensorBoard(log_dir=logdir)]  # TqdmCallback(verbose=2)
    start_time_epoch = time.time()
    # == Train ==
    history = model.fit(x=encoder_inputs, y=decoder_outputs, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                        verbose=2, callbacks=callbacks, validation_data=(enc_inputs_val, dec_outputs_val), shuffle=True)
    print("\ncompleted in {0:.2f} s".format((time.time() - start_time_epoch)))
    # Save the model
    print("\n==> Saving the model... ", end="")
    start_time = time.time()
    model.save(save_dir + 'baseline_model')
    print("\ndone in {0:.2f} ms".format(1000 * (time.time() - start_time)))

    # Reset global time and loss
    step_time, loss = 0, 0

    sys.stdout.flush()


if __name__ == "__main__":
    train()
