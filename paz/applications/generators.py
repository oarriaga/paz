import json
from pathlib import Path

import numpy as np
from keras import ops

from paz.models import Gemma4
from paz.models.foundation.gemma4.tokenizer import Gemma4Tokenizer
from paz.models.foundation.gemma4.image_converter import preprocess_images
from paz.models.foundation.gemma4.multimodal_decoding import generate_eager
from paz.models.foundation.gemma4.vision import VisionEncoderArgs


def GenerateGemma4(model_name="gemma4_2b", max_tokens=64, weights="paz",
                   models_path=None):
    models = Gemma4(model_name, weights=weights, models_path=models_path)
    tokenizer = build_gemma4_tokenizer(models_path)
    stop_id = tokenizer.get_stop_token_ids()[-1]
    no_vision = np.zeros((0, models.config.hidden_dim), "float32")

    def generate(prompt):
        token_ids = tokenizer.tokenize_generation_prompt(prompt)
        args = (models.decoder_step, models.per_layer_step, no_vision,
                models.config, token_ids, [], stop_id, max_tokens)
        return tokenizer.detokenize(generate_eager(*args))

    return generate


def DescribeImageGemma4(model_name="gemma4_2b", max_tokens=32, weights="paz",
                        models_path=None):
    models = Gemma4(model_name, weights=weights, models_path=models_path)
    tokenizer = build_gemma4_tokenizer(models_path)
    vision = VisionEncoderArgs(**read_vision_config(models_path))
    stop_id = tokenizer.get_stop_token_ids()[-1]

    def describe(image, question="Describe this image."):
        embeddings = encode_image(image, models, vision)
        token_ids, indices = build_image_prompt(
            tokenizer, len(embeddings), question)
        args = (models.decoder_step, models.per_layer_step, embeddings,
                models.config, token_ids, indices, stop_id, max_tokens)
        return tokenizer.detokenize(generate_eager(*args))

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
