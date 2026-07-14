import json
from pathlib import Path

import numpy as np
from keras import ops

from paz import place_on_model_device
from paz.models import Gemma4
from paz.models.foundation.gemma4.pretrained import resolve_dir
from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer
from paz.models.foundation.gemma4.image_converter import preprocess_images
from paz.models.foundation.gemma4.multimodal_decoding import (
    build_generator, build_text_generator)
from paz.models.foundation.gemma4.vision import VisionEncoderArgs


def GenerateGemma4(model_name="gemma4_2b", max_tokens=64, max_seq=512,
                   max_prompt=128, weights="pretrained", models_path=None,
                   models=None):
    model_dir = resolve_dir(model_name, models_path)
    if models is None:
        models = Gemma4(model_name, weights=weights, models_path=model_dir)
    tokenizer = build_gemma4_tokenizer(model_dir)
    stop_id = tokenizer.get_stop_token_ids()[-1]
    stream = build_token_printer(tokenizer, stop_id)
    args = (models.model, stop_id, max_tokens, max_seq, max_prompt)
    decode = build_text_generator(*args, emit=stream)

    def generate(prompt):
        token_ids = tokenizer.tokenize_generation_prompt(prompt)
        generated = decode(token_ids)
        print()
        return tokenizer.detokenize(generated)

    return generate


def build_token_printer(tokenizer, stop_id):
    def print_token(token_id):
        if int(token_id) != stop_id:
            print(tokenizer.detokenize([int(token_id)]), end="", flush=True)
    return print_token


def DescribeImageGemma4(model_name="gemma4_2b", max_tokens=64, max_seq=512,
                        max_prompt=400, weights="pretrained", models_path=None,
                        models=None):
    model_dir = resolve_dir(model_name, models_path)
    if models is None:
        models = Gemma4(model_name, weights=weights, models_path=model_dir)
    tokenizer = build_gemma4_tokenizer(model_dir)
    vision = VisionEncoderArgs(**read_vision_config(model_dir))
    stop_id = tokenizer.get_stop_token_ids()[-1]
    stream = build_token_printer(tokenizer, stop_id)
    args = (models.model, stop_id, max_tokens, max_seq, max_prompt)
    generate = build_generator(*args, emit=stream)

    def encode(image):
        return encode_image(image, models, vision)

    def answer(embeddings, question="Describe this image."):
        token_ids, indices = build_image_prompt(
            tokenizer, len(embeddings), question)
        generated = generate(token_ids, embeddings, indices)
        print()
        return tokenizer.detokenize(generated)

    def describe(image, question="Describe this image."):
        return answer(encode(image), question)

    # Expose the parts so callers can encode an image once and ask many
    # questions about it without re-running the vision encoder.
    describe.encode = encode
    describe.answer = answer
    return describe


def build_gemma4_tokenizer(models_path):
    return Gemma4Tokenizer(Path(models_path) / "tokenizer.json")


def read_vision_config(models_path):
    with open(str(Path(models_path) / "vision_config.json")) as file:
        return json.load(file)


def encode_image(image, models, vision):
    image = np.asarray(image, dtype="float32")
    if image.max() > 1.0:
        image = image / 255.0
    inputs = preprocess_images(image[None], vision)
    # Run the vision encoder wherever its weights live: placing the inputs on
    # that device lets the caller decide CPU vs GPU by how they built the model.
    inputs = place_on_model_device(inputs, models.vision_encoder)
    embeddings = np.asarray(models.vision_encoder(inputs))[0]
    positions = np.asarray(inputs["pixel_position_ids"])[0]
    pool = vision.pool_size
    width = (int(positions[:, 0].max()) + 1) // pool
    height = (int(positions[:, 1].max()) + 1) // pool
    valid = width * height
    scale = float(models.config.hidden_dim) ** -0.5
    embeddings = (embeddings[:valid] * scale).astype("float32")
    return ops.cast(embeddings, models.config.dtype)


def build_image_prompt(tokenizer, num_image_tokens, question):
    head = tokenizer.tokenize("<|turn>user\n")
    tail = tokenizer.tokenize("{}<turn|>\n<|turn>model\n".format(question))
    image_block = ([tokenizer.start_of_image_token_id]
                   + [tokenizer.image_placeholder_id] * num_image_tokens
                   + [tokenizer.end_of_image_token_id])
    token_ids = [tokenizer.start_token_id] + head + image_block + tail
    indices = []
    for index, token in enumerate(token_ids):
        if token == tokenizer.image_placeholder_id:
            indices.append(index)
    return token_ids, indices


def GenerateGemma42B(**kwargs):
    return GenerateGemma4("gemma4_2b", **kwargs)


def GenerateGemma44B(**kwargs):
    return GenerateGemma4("gemma4_4b", **kwargs)


def DescribeImageGemma42B(**kwargs):
    return DescribeImageGemma4("gemma4_2b", **kwargs)


def DescribeImageGemma44B(**kwargs):
    return DescribeImageGemma4("gemma4_4b", **kwargs)
