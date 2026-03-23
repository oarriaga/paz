import json
import os
import subprocess
import textwrap
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest

from paz.models.keypoint.detnet import DetNet
from paz.models.keypoint.iknet import IKNet


def build_inputs():
    random_state = np.random.default_rng(7)
    image = random_state.integers(
        0, 256, size=(1, 128, 128, 3), dtype=np.uint8
    )
    keypoints = random_state.normal(size=(1, 84, 3)).astype("float32")
    return image, keypoints


def run_compatibility_models():
    script = textwrap.dedent(
        """
        import json
        import os
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.initializers import TruncatedNormal
        from tensorflow.keras.initializers import VarianceScaling
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import Layer
        from tensorflow.keras.layers import MaxPool2D
        from tensorflow.keras.layers import ReLU
        from tensorflow.keras.layers import Reshape
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        def build_name(name):
            return name.replace("/", "_")

        class CastAndScale(Layer):
            def call(self, inputs):
                return tf.cast(inputs, tf.float32) / 255.0

            def compute_output_shape(self, input_shape):
                return input_shape

        class ZeroPadding(Layer):
            def __init__(self, pad_before, pad_after, **kwargs):
                super().__init__(**kwargs)
                self.pad_before = pad_before
                self.pad_after = pad_after

            def call(self, inputs):
                return tf.pad(
                    inputs,
                    [
                        [0, 0],
                        [self.pad_before, self.pad_after],
                        [self.pad_before, self.pad_after],
                        [0, 0],
                    ],
                )

            def compute_output_shape(self, input_shape):
                batch_size, height, width, channels = input_shape
                height = (
                    None
                    if height is None
                    else height + self.pad_before + self.pad_after
                )
                width = (
                    None
                    if width is None
                    else width + self.pad_before + self.pad_after
                )
                return (batch_size, height, width, channels)

        class PoseTile(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                axis = np.linspace(-1, 1, 32, dtype="float32")
                pose_grid = np.stack(
                    [
                        np.tile(axis.reshape(1, 32), [32, 1]),
                        np.tile(axis.reshape(32, 1), [1, 32]),
                    ],
                    axis=-1,
                )
                self.pose_grid = tf.constant(pose_grid[None, ...], dtype=tf.float32)

            def call(self, inputs):
                return tf.tile(self.pose_grid, [tf.shape(inputs)[0], 1, 1, 1])

            def compute_output_shape(self, input_shape):
                return (input_shape[0], 32, 32, 2)

        class HeatmapToUV(Layer):
            def call(self, inputs):
                shape = tf.shape(inputs)
                heatmap = tf.reshape(inputs, (shape[0], -1, shape[3]))
                argmax = tf.math.argmax(heatmap, axis=1, output_type=tf.int32)
                argmax_x = argmax // shape[2]
                argmax_y = argmax % shape[2]
                uv = tf.stack((argmax_x, argmax_y), axis=1)
                return tf.transpose(a=uv, perm=[0, 2, 1])

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[3], 2)

        class GatherXYZ(Layer):
            def call(self, inputs):
                location_map, uv = inputs
                return tf.gather_nd(
                    tf.transpose(location_map, perm=[0, 3, 1, 2, 4]),
                    uv,
                    batch_dims=2,
                )

            def compute_output_shape(self, input_shape):
                location_shape, _ = input_shape
                return (location_shape[0], location_shape[3], 3)

        class TakeFirst(Layer):
            def call(self, inputs):
                return inputs[0]

            def compute_output_shape(self, input_shape):
                return input_shape[1:]

        class Normalize(Layer):
            def call(self, inputs):
                values_norm = tf.sqrt(
                    tf.reduce_sum(inputs * inputs, axis=-1, keepdims=True)
                )
                return inputs / tf.maximum(values_norm, 1e-6)

            def compute_output_shape(self, input_shape):
                return input_shape

        class MakePositive(Layer):
            def call(self, inputs):
                positive_mask = tf.tile(inputs[:, :, 0:1] > 0, [1, 1, 4])
                return tf.where(positive_mask, inputs, -inputs)

            def compute_output_shape(self, input_shape):
                return input_shape

        class ReorderQuaternions(Layer):
            def call(self, inputs):
                scalar = inputs[:, :, 0:1]
                vector = inputs[:, :, 1:4]
                return tf.concat((vector, scalar), axis=-1)

            def compute_output_shape(self, input_shape):
                return input_shape

        def build_conv_block(
            tensor, filters, kernel_size, strides, name, rate=1, with_relu=True
        ):
            if strides == 1:
                tensor = Conv2D(
                    filters,
                    kernel_size,
                    strides,
                    padding="same",
                    use_bias=False,
                    dilation_rate=rate,
                    kernel_regularizer=l2(0.5),
                    name=build_name(name + "/conv2d"),
                    kernel_initializer=VarianceScaling(
                        mode="fan_avg", distribution="uniform"
                    ),
                )(tensor)
            else:
                pad_before = (kernel_size - 1) // 2
                pad_after = (kernel_size - 1) - pad_before
                tensor = ZeroPadding(pad_before, pad_after)(tensor)
                tensor = Conv2D(
                    filters,
                    kernel_size,
                    strides,
                    padding="valid",
                    use_bias=False,
                    dilation_rate=rate,
                    kernel_regularizer=l2(0.5),
                    name=build_name(name + "/conv2d"),
                    kernel_initializer=VarianceScaling(
                        mode="fan_avg", distribution="uniform"
                    ),
                )(tensor)
            tensor = BatchNormalization(
                name=build_name(name + "/batch_normalization")
            )(tensor)
            if with_relu:
                tensor = ReLU(name=build_name(name + "/activation"))(tensor)
            return tensor

        def build_bottleneck(tensor, filters, strides, name, rate=1):
            num_channels = tensor.shape[-1]
            if num_channels == filters:
                if strides == 1:
                    shortcut = tensor
                else:
                    shortcut = MaxPool2D(
                        strides,
                        strides,
                        "same",
                        name=build_name(name + "/shortcut/max_pool"),
                    )(tensor)
            else:
                shortcut = build_conv_block(
                    tensor,
                    filters,
                    1,
                    strides,
                    name + "/shortcut",
                    with_relu=False,
                )
            residual = build_conv_block(
                tensor, filters // 4, 1, 1, name + "/conv1"
            )
            residual = build_conv_block(
                residual, filters // 4, 3, strides, name + "/conv2", rate
            )
            residual = build_conv_block(
                residual, filters, 1, 1, name + "/conv3", with_relu=False
            )
            return ReLU(name=build_name(name + "/relu"))(shortcut + residual)

        def build_resnet50(tensor, name):
            tensor = build_conv_block(tensor, 64, 7, 2, name + "/conv1")
            for unit_arg in range(2):
                tensor = build_bottleneck(
                    tensor, 256, 1, name + f"/block1/unit{unit_arg + 1}"
                )
            tensor = build_bottleneck(tensor, 256, 2, name + "/block1/unit3")
            for unit_arg in range(4):
                tensor = build_bottleneck(
                    tensor, 512, 1, name + f"/block2/unit{unit_arg + 1}", 2
                )
            for unit_arg in range(6):
                tensor = build_bottleneck(
                    tensor, 1024, 1, name + f"/block3/unit{unit_arg + 1}", 4
                )
            return build_conv_block(tensor, 256, 3, 1, name + "/squeeze")

        def build_2d_head(features, num_keypoints, name):
            projected = build_conv_block(features, 256, 3, 1, name + "/project")
            return Conv2D(
                num_keypoints,
                1,
                strides=1,
                padding="same",
                activation="sigmoid",
                name=build_name(name + "/prediction/conv2d"),
                kernel_initializer=TruncatedNormal(stddev=0.01),
            )(projected)

        def build_3d_head(features, num_keypoints, name):
            projected = build_conv_block(features, 256, 3, 1, name + "/project")
            projected = Conv2D(
                num_keypoints * 3,
                1,
                strides=1,
                padding="same",
                name=build_name(name + "/prediction/conv2d"),
                kernel_initializer=TruncatedNormal(stddev=0.01),
            )(projected)
            height, width = features.shape[1:3]
            return Reshape(
                [height, width, num_keypoints, 3],
                name=build_name(name + "/prediction/reshape"),
            )(projected)

        def build_dense(features, num_units, name):
            return Dense(
                num_units,
                activation=None,
                kernel_regularizer=l2(0.5),
                kernel_initializer=TruncatedNormal(stddev=0.01),
                name=name,
            )(features)

        def build_dense_block(features, num_units, depth_arg):
            features = build_dense(
                features, num_units, f"dense_block_{depth_arg}_dense"
            )
            return BatchNormalization(
                name=f"batch_normalization_{depth_arg}"
            )(features)

        def build_detnet():
            image = Input(shape=(128, 128, 3), dtype=tf.uint8)
            features = CastAndScale()(image)
            name = "prior_based_hand"
            features = build_resnet50(features, name + "/resnet")
            pose_tile = PoseTile(name=build_name(name + "/pose_tile"))(features)
            features = Concatenate(
                axis=-1, name=build_name(name + "/pose_concat")
            )([features, pose_tile])
            heat_map = build_2d_head(features, 21, name + "/hmap_0")
            features = Concatenate(
                axis=-1, name=build_name(name + "/heat_map_concat")
            )([features, heat_map])
            delta_map = build_3d_head(features, 21, name + "/dmap_0")
            reshaped_delta_map = Reshape(
                [32, 32, 21 * 3], name=build_name(name + "/reshaped_delta_map")
            )(delta_map)
            features = Concatenate(
                axis=-1, name=build_name(name + "/delta_map_concat")
            )([features, reshaped_delta_map])
            location_map = build_3d_head(features, 21, name + "/lmap_0")
            reshaped_location_map = Reshape(
                [32, 32, 21 * 3],
                name=build_name(name + "/reshaped_location_map"),
            )(location_map)
            features = Concatenate(
                axis=-1, name=build_name(name + "/location_map_concat")
            )([features, reshaped_location_map])
            uv = HeatmapToUV()(heat_map)
            xyz = GatherXYZ()([location_map, uv])
            xyz = TakeFirst()(xyz)
            uv = TakeFirst()(uv)
            return Model(image, [xyz, uv], name="detnet")

        def build_iknet():
            keypoints = Input(shape=(84, 3), dtype=tf.float32, name="keypoints")
            features = Reshape([1, -1], name="input_reshape")(keypoints)
            for depth_arg in range(6):
                features = build_dense_block(features, 1024, depth_arg)
                features = Activation(
                    "sigmoid", name=f"sigmoid_activation_{depth_arg}"
                )(features)
            features = build_dense(features, 21 * 4, "output_dense")
            features = Reshape([21, 4], name="output_reshape")(features)
            features = Normalize()(features)
            features = MakePositive()(features)
            features = ReorderQuaternions()(features)
            return Model(keypoints, features, name="iknet")

        random_state = np.random.default_rng(7)
        image = random_state.integers(
            0, 256, size=(1, 128, 128, 3), dtype=np.uint8
        )
        keypoints = random_state.normal(size=(1, 84, 3)).astype("float32")

        detnet = build_detnet()
        detnet.load_weights(os.path.expanduser("~/.keras/paz/models/detnet_weights.hdf5"))
        detnet_xyz, detnet_uv = detnet(image)

        iknet = build_iknet()
        iknet.load_weights(os.path.expanduser("~/.keras/paz/models/iknet_weight.hdf5"))
        iknet_quaternions = iknet(keypoints)

        outputs = {
            "detnet_xyz": np.asarray(detnet_xyz).tolist(),
            "detnet_uv": np.asarray(detnet_uv).tolist(),
            "iknet": np.asarray(iknet_quaternions).tolist(),
        }
        print(json.dumps(outputs))
        """
    )
    env = dict(os.environ)
    env.pop("KERAS_BACKEND", None)
    output = subprocess.check_output(
        ["python", "-c", script], text=True, env=env
    ).strip()
    return json.loads(output.splitlines()[-1])


@pytest.fixture(scope="module")
def compatibility_outputs():
    return run_compatibility_models()


def test_detnet_invalid_weights():
    with pytest.raises(ValueError):
        DetNet(weights="invalid")


def test_iknet_invalid_weights():
    with pytest.raises(ValueError):
        IKNet(weights="invalid")


def test_detnet_pretrained_weights_load():
    model = DetNet(weights="detnet")
    image, _ = build_inputs()
    xyz, uv = model(image)
    assert tuple(np.asarray(xyz).shape) == (21, 3)
    assert tuple(np.asarray(uv).shape) == (21, 2)


def test_iknet_pretrained_weights_load():
    model = IKNet(weights="iknet")
    _, keypoints = build_inputs()
    quaternions = model(keypoints)
    assert tuple(np.asarray(quaternions).shape) == (1, 21, 4)


def test_detnet_pretrained_matches_compatibility_model(compatibility_outputs):
    image, _ = build_inputs()
    model = DetNet(weights="detnet")
    xyz, uv = model(image)
    assert np.allclose(
        np.asarray(xyz), compatibility_outputs["detnet_xyz"], atol=1e-6
    )
    assert np.array_equal(np.asarray(uv), compatibility_outputs["detnet_uv"])


def test_iknet_pretrained_matches_compatibility_model(compatibility_outputs):
    _, keypoints = build_inputs()
    model = IKNet(weights="iknet")
    quaternions = model(keypoints)
    assert np.allclose(
        np.asarray(quaternions), compatibility_outputs["iknet"], atol=1e-6
    )


def test_detnet_port_weights_script(tmp_path):
    output_path = tmp_path / "detnet_paz_jax.weights.h5"
    script_path = Path("paz/models/keypoint/port_detnet_weights.py")
    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    subprocess.check_output(
        [
            "python",
            str(script_path),
            "--output_path",
            str(output_path),
        ],
        text=True,
        env=env,
    )

    image, _ = build_inputs()
    pretrained_model = DetNet(weights="detnet")
    expected_xyz, expected_uv = pretrained_model(image)

    ported_model = DetNet(weights=None)
    ported_model.load_weights(output_path)
    xyz, uv = ported_model(image)

    assert np.allclose(np.asarray(xyz), np.asarray(expected_xyz), atol=1e-6)
    assert np.array_equal(np.asarray(uv), np.asarray(expected_uv))
