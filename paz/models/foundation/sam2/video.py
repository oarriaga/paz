"""Streaming video tracking with SAM 2: prompt any frame, get every frame.

``track`` prompts one or more objects on one or more frames and then walks the
video forward once. A frame is encoded by the image encoder, conditioned on the
memory bank, decoded by the SAM heads, and encoded back into memory. The bank
holds the prompted frames, the last six tracked frames, and up to sixteen
object pointers. Each object keeps its own bank, mirroring the official
predictor's per-object slices; frames are shared, so the image encoder runs
once per frame. The bank is padded to its maximum size and carries a keep-mask,
so every frame feeds the same shapes and each sub-model compiles once.

Yielded masks are logits at the video resolution: positive inside the object,
``NO_OBJECT`` on frames where the model predicts it is gone. Memories are
rounded to bfloat16 before storage, as the official predictor does. Not
implemented: correcting a prompt after tracking, tracking backwards, and the
optional filling of small holes.
"""
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jp

import paz
from paz.models.foundation.sam2 import memory_attention, memory_encoder
from paz.models.foundation.sam2.configuration import IMAGE_SIZE, MEMORY_DIM
from paz.models.foundation.sam2.configuration import NUM_MEMORIES
from paz.models.foundation.sam2.configuration import PROMPT_EMBED_DIM
from paz.models.foundation.sam2.predict import build_dense, build_prompt
from paz.models.foundation.sam2.preprocessing import preprocess_image
from paz.models.foundation.sam2.prompt_encoder import GRID, MASK_INPUT
from paz.models.foundation.sam2.prompt_encoder import dense_positional_encoding

NUM_POINTERS = 16
POINTER_SPLIT = PROMPT_EMBED_DIM // MEMORY_DIM
NO_OBJECT = -1024.0
MASK_SCALE = 20.0
MASK_BIAS = -10.0
STABILITY_DELTA = 0.05
STABILITY_LIMIT = 0.98

PROMPT = "frame object_id points labels box mask"
Prompt = namedtuple("Prompt", PROMPT, defaults=(None,) * 4)
Frame = namedtuple("Frame", "embedding features high_res_0 high_res_1")
Entry = namedtuple("Entry", "memory pointer masks")
Track = namedtuple("Track", "prompted tracked")
Decoded = namedtuple("Decoded", "masks token absent")
Bank = namedtuple("Bank", "memory positions times mask")
CONSTANTS = "image_pe frame_pos memory_pos rotary prompt spatial pointers"
Constants = namedtuple("Constants", CONSTANTS)


def track(bundle, images, prompts):
    constants = build_constants(bundle, count_prompted_frames(prompts))
    size = images[0].shape[:2]
    tracks = start_tracks(bundle, images, prompts, constants, size)
    objects = sorted(tracks)
    for frame in range(min(prompt.frame for prompt in prompts), len(images)):
        pending = select_pending(tracks, objects, frame)
        features = encode_frame(bundle, images[frame]) if pending else None
        for name in pending:
            arguments = bundle, features, constants, tracks[name]
            tracks[name].tracked[frame] = follow(*arguments, frame, len(images))
        masks = [read_entry(tracks[name], frame).masks for name in objects]
        yield frame, resize_masks(jp.concatenate(masks), size)


def select_pending(tracks, objects, frame):
    return [name for name in objects if frame not in tracks[name].prompted]


def count_prompted_frames(prompts):
    # Every prompted frame stays in the bank, so the most-prompted object sets
    # how large the padded bank has to be for the whole video.
    counts = {}
    for prompt in prompts:
        counts[prompt.object_id] = counts.get(prompt.object_id, 0) + 1
    return max(counts.values())


def start_tracks(bundle, images, prompts, constants, size):
    tracks = {}
    for frame in sorted({prompt.frame for prompt in prompts}):
        features = encode_frame(bundle, images[frame])
        for prompt in [entry for entry in prompts if entry.frame == frame]:
            track = tracks.setdefault(prompt.object_id, Track({}, {}))
            arguments = bundle, features, constants, prompt, size
            track.prompted[frame] = begin(*arguments)
    return tracks


def begin(bundle, features, constants, prompt, size):
    if prompt.mask is None:
        entry = begin_from_points(bundle, features, constants, prompt, size)
    else:
        entry = begin_from_mask(bundle, features, constants, prompt.mask)
    return entry


def begin_from_points(bundle, features, constants, prompt, size):
    coordinates, labels = build_prompt(*unpack_prompt(prompt), size)
    sparse = bundle.point_encoder((coordinates, labels))
    prompts = sparse, build_dense(bundle, None)
    select = select_best if count_points(prompt) <= 1 else select_stable
    arguments = bundle, features, features.embedding, prompts
    decoded = decode(*arguments, constants.image_pe, select)
    binary = binarize(decoded.masks)
    return build_entry(bundle, features, decoded, binary, decoded.absent)


def begin_from_mask(bundle, features, constants, mask):
    prompt = resize_prompt(mask)
    absent = jp.asarray(jp.max(prompt) <= 0.0, jp.float32).reshape(1, 1)
    decoded = decode_mask_prompt(bundle, features, constants, prompt)
    masks = downscale_mask(prompt * MASK_SCALE + MASK_BIAS)
    # The mask, not the decoder, decides whether the object is here; an empty
    # mask also overrides the pointer the decoder read off the prompt.
    seeded = Decoded(masks, decoded.token, jp.maximum(decoded.absent, absent))
    return build_entry(bundle, features, seeded, binarize(masks), absent)


def follow(bundle, features, constants, track, frame, num_frames):
    embedding = condition(bundle, features, constants, track, frame, num_frames)
    arguments = bundle, features, embedding, constants.prompt
    decoded = decode(*arguments, constants.image_pe, select_best)
    smoothed = jax.nn.sigmoid(upscale_mask(decoded.masks))
    return build_entry(bundle, features, decoded, smoothed, decoded.absent)


def build_entry(bundle, features, decoded, probabilities, absent):
    mask = probabilities * MASK_SCALE + MASK_BIAS
    memory = bundle.memory_encoder((features.features, mask, absent))
    pointer = bundle.pointer((decoded.token, decoded.absent))
    return Entry(to_bfloat16(memory), pointer, decoded.masks)


def decode(bundle, features, embedding, prompts, image_pe, select):
    sparse, dense = prompts
    tensors = (embedding, features.high_res_0, features.high_res_1)
    inputs = (*tensors, sparse, dense, image_pe)
    masks, scores, objectness, tokens = bundle.mask_decoder(inputs)
    masks, token = select(masks, scores, tokens)
    absent = jp.asarray(objectness <= 0.0, jp.float32)
    gated = jp.where(absent[:, :, None] > 0.0, NO_OBJECT, masks)
    return Decoded(gated, token, absent)


def decode_mask_prompt(bundle, features, constants, prompt):
    low_res = bundle.mask_downsample(prompt)
    sparse, _ = constants.prompt
    prompts = sparse, build_dense(bundle, low_res)
    arguments = bundle, features, features.features, prompts
    return decode(*arguments, constants.image_pe, select_stable)


def select_best(masks, scores, tokens):
    best = 1 + int(jp.argmax(scores[0, 1:]))
    return masks[:, best], tokens[:, best]


def select_stable(masks, scores, tokens):
    best = 1 + int(jp.argmax(scores[0, 1:]))
    index = 0 if stability(masks[0, 0]) >= STABILITY_LIMIT else best
    return masks[:, index], tokens[:, 0]


def stability(mask):
    inner = jp.sum(mask > STABILITY_DELTA)
    outer = jp.sum(mask > -STABILITY_DELTA)
    return jp.where(outer > 0, inner / outer, 1.0)


def condition(bundle, features, constants, track, frame, num_frames):
    arguments = bundle, features, constants, track
    inputs = build_memory_inputs(*arguments, frame, num_frames)
    tokens = bundle.memory_attention(inputs)
    return jp.reshape(tokens, (1, GRID, GRID, PROMPT_EMBED_DIM))


def build_memory_inputs(bundle, features, constants, track, frame, num_frames):
    spatial = build_spatial(track, frame, constants)
    found = bundle, track, frame, num_frames, constants.pointers
    banked = join_banks(spatial, build_pointers(*found))
    current = flatten(features.features), constants.frame_pos
    return current + tuple(banked) + constants.rotary


def join_banks(spatial, pointers):
    joined = []
    for parts in zip(spatial, pointers):
        joined.append(jp.concatenate(parts, axis=1))
    return Bank(*joined)


def build_spatial(track, frame, constants):
    entries, slots = select_memories(track, frame)
    memories = [flatten(entry.memory) for entry in entries]
    memory = pad_tokens(jp.concatenate(memories, axis=1), constants.spatial)
    times = pad_tokens(build_times(slots), constants.spatial)
    mask = build_mask(len(entries) * GRID * GRID, constants.spatial)
    return Bank(memory, constants.memory_pos, times, mask)


def select_memories(track, frame):
    entries = [track.prompted[index] for index in sorted(track.prompted)]
    slots = [NUM_MEMORIES - 1] * len(entries)
    for distance in reversed(range(1, NUM_MEMORIES)):
        previous = track.tracked.get(frame - distance)
        if previous is not None:
            entries.append(previous)
            slots.append(distance - 1)
    return entries, slots


def build_pointers(bundle, track, frame, num_frames, length):
    distances, pointers = select_pointers(track, frame, num_frames)
    if pointers:
        found = bundle, distances, pointers, num_frames, length
        tokens, positions = encode_pointers(*found)
    else:
        tokens = positions = jp.zeros((1, 0, MEMORY_DIM))
    mask = build_mask(tokens.shape[1], length)
    times = jp.zeros((1, length, NUM_MEMORIES))
    padded = pad_tokens(tokens, length), pad_tokens(positions, length)
    return Bank(*padded, times, mask)


def pad_tokens(tokens, length):
    missing = length - tokens.shape[1]
    return jp.pad(tokens, ((0, 0), (0, missing), (0, 0)))


def build_mask(count, length):
    return (jp.arange(length) < count).astype(jp.float32)[None]


def select_pointers(track, frame, num_frames):
    distances, pointers = [], []
    for index in sorted(track.prompted):
        if index <= frame:
            distances.append(frame - index)
            pointers.append(track.prompted[index].pointer)
    for distance in range(1, min(num_frames, NUM_POINTERS)):
        if frame - distance < 0:
            break
        previous = track.tracked.get(frame - distance)
        if previous is not None:
            distances.append(distance)
            pointers.append(previous.pointer)
    return distances, pointers


def encode_pointers(bundle, distances, pointers, num_frames, length):
    limit = min(num_frames, NUM_POINTERS) - 1
    scaled = np.zeros(length // POINTER_SPLIT, np.float32)
    scaled[:len(distances)] = np.asarray(distances, np.float32) / limit
    positions = jp.asarray(bundle.pointer_time(sine_encoding(scaled)))
    tokens = jp.concatenate(pointers, axis=0)
    repeated = jp.repeat(positions, POINTER_SPLIT, axis=0)
    return jp.reshape(tokens, (1, -1, MEMORY_DIM)), repeated[None]


def sine_encoding(distances, dim=PROMPT_EMBED_DIM, temperature=10000.0):
    half = dim // 2
    powers = 2.0 * (np.arange(half, dtype=np.float32) // 2) / half
    angles = distances[:, None] / temperature ** powers
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def build_times(slots):
    rows = jax.nn.one_hot(jp.asarray(slots), NUM_MEMORIES)
    return jp.repeat(rows, GRID * GRID, axis=0)[None]


def build_constants(bundle, num_prompted):
    entries = num_prompted + NUM_MEMORIES - 1
    spatial = entries * GRID * GRID
    pointers = (num_prompted + NUM_POINTERS - 1) * POINTER_SPLIT
    tables = build_rotary(entries, pointers), build_empty_prompt(bundle)
    sizes = spatial, pointers
    return Constants(*build_positions(bundle, entries), *tables, *sizes)


def build_positions(bundle, entries):
    image_pe = dense_positional_encoding(bundle.point_encoder)[None]
    frame_pos = sine_positions(PROMPT_EMBED_DIM)
    memory_pos = jp.tile(sine_positions(MEMORY_DIM), (1, entries, 1))
    return image_pe, frame_pos, memory_pos


def build_rotary(entries, pointers):
    cos, sin = memory_attention.rotary_tables(GRID, GRID)
    identity = memory_attention.identity_tables(pointers)
    memory_cos = tile_rotary(cos, identity[0], entries)
    memory_sin = tile_rotary(sin, identity[1], entries)
    return jp.asarray(cos)[None], jp.asarray(sin)[None], memory_cos, memory_sin


def tile_rotary(table, pointers, entries):
    tiled = np.tile(table, (entries, 1))
    return jp.asarray(np.concatenate([tiled, pointers], axis=0))[None]


def sine_positions(num_features):
    encoding = memory_encoder.sine_position_encoding(GRID, GRID, num_features)
    return flatten(jp.asarray(encoding))


def build_empty_prompt(bundle):
    coordinates = jp.zeros((1, 2, 2), jp.float32)
    labels = -jp.ones((1, 2), jp.float32)
    sparse = bundle.point_encoder((coordinates, labels))
    return sparse, build_dense(bundle, None)


def encode_frame(bundle, image):
    pixels = preprocess_image(image)[None]
    outputs = bundle.image_encoder(pixels)
    embedding, high_res_0, high_res_1, features = outputs
    return Frame(embedding, features, high_res_0, high_res_1)


def read_entry(track, frame):
    if frame in track.prompted:
        entry = track.prompted[frame]
    else:
        entry = track.tracked[frame]
    return entry


def unpack_prompt(prompt):
    return prompt.points, prompt.labels, prompt.box


def count_points(prompt):
    corners = 0 if prompt.box is None else 2
    points = 0 if prompt.points is None else len(prompt.points)
    return corners + points


def binarize(masks):
    return jp.asarray(upscale_mask(masks) > 0.0, jp.float32)


def upscale_mask(masks):
    channels_last = jp.transpose(masks, (1, 2, 0))
    size = (IMAGE_SIZE, IMAGE_SIZE)
    return paz.image.resize(channels_last, size, "linear", False)[None]


def downscale_mask(logits):
    size = (MASK_INPUT, MASK_INPUT)
    resized = paz.image.resize(logits[0], size, "linear", True)
    return jp.transpose(resized, (2, 0, 1))


def resize_prompt(mask):
    values = jp.asarray(mask, jp.float32)[..., None]
    size = (IMAGE_SIZE, IMAGE_SIZE)
    resized = paz.image.resize(values, size, "linear", True)
    return jp.asarray(resized >= 0.5, jp.float32)[None]


def resize_masks(masks, size):
    channels_last = jp.transpose(masks, (1, 2, 0))
    resized = paz.image.resize(channels_last, size, "linear", False)
    return jp.transpose(resized, (2, 0, 1))


def flatten(x):
    return jp.reshape(x, (1, GRID * GRID, x.shape[-1]))


def to_bfloat16(memory):
    return jp.asarray(jp.asarray(memory, jp.bfloat16), jp.float32)
