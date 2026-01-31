import argparse
import os

import numpy as np
from keras import ops
from keras.utils import img_to_array
from keras.utils import load_img

from examples.gemma3.functional.gemma3 import apply_reversible_projection
from examples.gemma3.functional.keras_hub_utils import ensure_keras_hub
from examples.gemma3.functional.weights import load_gemma3_backbone_from_preset


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--preset", default="gemma3_270m")
    parser.add_argument("--sequence-length", type=int, default=None)
    return parser.parse_args()


def _load_image(image_path):
    image = load_img(image_path)
    return img_to_array(image)


def _to_numpy(value):
    try:
        return ops.convert_to_numpy(value)
    except Exception:
        return np.array(value)


def _decode_token(tokenizer, token_id):
    ids = np.array([[token_id]], dtype=np.int32)
    text = tokenizer.detokenize(ids)
    text = _to_numpy(text)
    if isinstance(text, np.ndarray):
        text = text[0]
    if isinstance(text, bytes):
        return text.decode("utf-8")
    if hasattr(text, "decode"):
        return text.decode("utf-8")
    return str(text)


def main():
    args = _parse_args()
    os.environ.setdefault("KERAS_BACKEND", "jax")
    keras_hub = ensure_keras_hub()

    if "<start_of_image>" not in args.prompt:
        message = "Prompt must include <start_of_image> for image inputs."
        raise ValueError(message)

    preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
        args.preset
    )
    image = _load_image(args.image)
    inputs = {"prompts": args.prompt, "images": image}
    processed = preprocessor.generate_preprocess(
        inputs, sequence_length=args.sequence_length
    )

    token_ids = _to_numpy(processed["token_ids"])
    padding_mask = _to_numpy(processed["padding_mask"])
    images = _to_numpy(processed["images"])
    vision_indices = _to_numpy(processed["vision_indices"])
    vision_mask = _to_numpy(processed["vision_mask"])

    result = load_gemma3_backbone_from_preset(
        preset=args.preset,
        dtype="float32",
        load_weights=True,
        sequence_length=args.sequence_length,
        num_images=1,
        batch_size=1,
        return_hub_model=False,
    )
    apply_backbone = result["apply_backbone"]
    layers = result["layers"]
    token_embedding = layers[0]
    logit_soft_cap = result["backbone_config"]["final_logit_soft_cap"]

    hidden = apply_backbone(
        token_ids,
        padding_mask,
        images,
        vision_indices,
        vision_mask,
        False,
    )
    logits = apply_reversible_projection(
        token_embedding, hidden, logit_soft_cap
    )

    last_index = int(np.sum(padding_mask[0]) - 1)
    last_logits = logits[:, last_index, :]
    next_id = int(_to_numpy(ops.argmax(last_logits, axis=-1))[0])
    next_text = _decode_token(preprocessor.tokenizer, next_id)

    print("Prompt:", args.prompt)
    print("Next token id:", next_id)
    print("Next token text:", next_text)


if __name__ == "__main__":
    main()
