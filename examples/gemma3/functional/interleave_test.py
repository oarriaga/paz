import numpy as np
from keras import ops

from examples.gemma3.functional.gemma3 import interleave_embeddings
from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub

ensure_keras_hub()

from keras_hub.src.models.gemma3 import gemma3_interleave_embeddings


def test_interleave_embeddings_matches_hub():
    rng = np.random.default_rng(4)
    batch_size = 2
    sequence_length = 6
    embedding_dim = 4
    num_vision_tokens = 2

    text_shape = (batch_size, sequence_length, embedding_dim)
    text_values = rng.standard_normal(text_shape).astype("float32")
    text_embeddings = ops.convert_to_tensor(text_values)

    image_shape = (batch_size, num_vision_tokens, embedding_dim)
    image_values = rng.standard_normal(image_shape).astype("float32")
    image_embeddings = ops.convert_to_tensor(image_values)

    vision_indices = ops.convert_to_tensor([[2, 3], [1, 2]], dtype="int32")

    interleave = interleave_embeddings
    hub_class = gemma3_interleave_embeddings.Gemma3InterleaveEmbeddings
    hub_layer = hub_class(num_vision_tokens)

    img = image_embeddings
    txt = text_embeddings
    idx = vision_indices
    num_tokens = num_vision_tokens
    hub_output = hub_layer(img, txt, idx)
    clean_output = interleave(img, txt, idx, num_tokens)
    clean_np = ops.convert_to_numpy(clean_output)
    hub_np = ops.convert_to_numpy(hub_output)
    np.testing.assert_allclose(clean_np, hub_np, rtol=1e-6, atol=1e-6)
