"""Video memory-module and tracker tests without torch, weights, or internet.

The modules are checked for shape and for the parameter-free encodings; the
tracker is checked on the parts that decide correctness: which memories and
pointers a frame attends to, how they are laid out, and how a mask is chosen.
"""
import numpy as np

from paz.models.foundation.sam2 import memory_encoder as me
from paz.models.foundation.sam2 import memory_attention as ma
from paz.models.foundation.sam2 import model as sam2_model
from paz.models.foundation.sam2 import pointer, video
from paz.models.foundation.sam2.configuration import TINY

GRID = video.GRID


def test_memory_encoder_output_shape():
    model = me.build()
    pix_feat = np.zeros((1, 64, 64, 256), np.float32)
    mask = np.zeros((1, 1024, 1024, 1), np.float32)
    absent = np.zeros((1, 1), np.float32)
    features = np.array(model((pix_feat, mask, absent)))
    assert features.shape == (1, 64, 64, 64)


def test_memory_encoder_adds_absent_embedding():
    model = me.build()
    layer = model.get_layer("no_obj_embed_spatial")
    layer.set_weights([np.full((1, 64), 3.0, "float32")])
    pix_feat = np.zeros((1, 64, 64, 256), np.float32)
    mask = np.zeros((1, 1024, 1024, 1), np.float32)
    present = np.array(model((pix_feat, mask, np.zeros((1, 1), np.float32))))
    absent = np.array(model((pix_feat, mask, np.ones((1, 1), np.float32))))
    assert np.allclose(absent - present, 3.0, atol=1e-5)


def test_sine_position_encoding_shape():
    encoding = me.sine_position_encoding(64, 64, 64)
    assert encoding.shape == (1, 64, 64, 64)


def test_rotary_tables_shape():
    cos, sin = ma.rotary_tables(4, 4)
    assert cos.shape == (16, ma.ROPE_DIM)
    assert sin.shape == (16, ma.ROPE_DIM)


def test_identity_tables_are_neutral():
    cos, sin = ma.identity_tables(5)
    assert np.allclose(cos, 1.0)
    assert np.allclose(sin, 0.0)


def memory_attention_inputs(spatial, pointers):
    total = spatial * spatial + pointers
    curr = np.zeros((1, spatial * spatial, 256), np.float32)
    memory = np.zeros((1, total, 64), np.float32)
    times = np.zeros((1, total, 7), np.float32)
    cos, sin = ma.rotary_tables(spatial, spatial)
    identity_cos, identity_sin = ma.identity_tables(pointers)
    memory_cos = np.concatenate([cos, identity_cos], axis=0)[None]
    memory_sin = np.concatenate([sin, identity_sin], axis=0)[None]
    rope = [cos[None], sin[None], memory_cos, memory_sin]
    return [curr, curr, memory, memory, times] + rope


def test_memory_attention_output_shape():
    tokens = np.array(ma.build()(memory_attention_inputs(4, 2)))
    assert tokens.shape == (1, 16, 256)


def test_temporal_encoding_selects_a_table_row():
    model = ma.build()
    layer = model.get_layer("maskmem_tpos_enc")
    table = np.arange(7 * 64, dtype="float32").reshape(7, 64)
    layer.set_weights([table])
    row = np.array(layer(np.eye(7, dtype="float32")[[3]]))
    assert np.allclose(row[0], table[3])


def test_pointer_falls_back_to_no_object():
    model = pointer.build()
    model.get_layer("no_obj_ptr").set_weights([np.full((1, 256), 5.0, "f4")])
    token = np.ones((1, 256), np.float32)
    absent = np.array(model((token, np.ones((1, 1), np.float32))))
    assert np.allclose(absent, 5.0, atol=1e-5)


def test_count_points_counts_box_corners():
    assert video.count_points(video.Prompt(0, 1, box=(0, 0, 1, 1))) == 2
    prompt = video.Prompt(0, 1, points=[(1, 2)], labels=[1])
    assert video.count_points(prompt) == 1


def build_track(prompted, tracked):
    entries = {frame: entry(frame) for frame in prompted}
    followed = {frame: entry(frame) for frame in tracked}
    return video.Track(entries, followed)


def entry(frame):
    memory = np.full((1, GRID, GRID, 64), float(frame), np.float32)
    return video.Entry(memory, np.full((1, 256), float(frame)), None)


def test_select_memories_orders_prompts_then_recent():
    track = build_track([0], [4, 6, 7, 8, 9])
    entries, slots = video.select_memories(track, 10)
    frames = [float(item.memory[0, 0, 0, 0]) for item in entries]
    assert frames == [0.0, 4.0, 6.0, 7.0, 8.0, 9.0]
    assert slots == [6, 5, 3, 2, 1, 0]


def test_select_memories_slots_count_distance():
    track = build_track([0], [7, 8, 9])
    _, slots = video.select_memories(track, 10)
    assert slots == [6, 2, 1, 0]


def test_select_memories_skips_frames_outside_the_window():
    track = build_track([0], [2, 9])
    entries, slots = video.select_memories(track, 10)
    assert len(entries) == 2
    assert slots == [6, 0]


def test_select_pointers_uses_past_only():
    track = build_track([0, 8], [1, 2])
    distances, pointers = video.select_pointers(track, 3, 6)
    assert distances == [3, 1, 2]
    assert len(pointers) == 3


def test_select_pointers_stops_at_the_first_frame():
    track = build_track([0], [])
    distances, _ = video.select_pointers(track, 0, 6)
    assert distances == [0]


def test_build_times_is_one_hot_per_memory_frame():
    times = np.array(video.build_times([6, 1], 8))
    assert times.shape == (1, 2 * GRID * GRID + 8, video.NUM_MEMORIES)
    assert np.allclose(times[0, 0], np.eye(7)[6])
    assert np.allclose(times[0, GRID * GRID], np.eye(7)[1])
    assert np.allclose(times[0, -8:], 0.0)


def test_memory_rotary_is_neutral_for_pointers():
    shape = (1, 4, ma.ROPE_DIM)
    tables = np.ones(shape), np.zeros(shape)
    constants = video.Constants(None, None, None, *tables, None)
    cos, sin = video.memory_rotary(constants, 2, 3)
    assert cos.shape == (1, 11, ma.ROPE_DIM)
    assert np.allclose(np.array(cos)[0, -3:], 1.0)
    assert np.allclose(np.array(sin)[0, -3:], 0.0)


def test_sine_encoding_is_bounded():
    encoding = video.sine_encoding(np.array([0.0, 0.5, 1.0], np.float32))
    assert encoding.shape == (3, 256)
    assert np.abs(encoding).max() <= 1.0
    assert np.allclose(encoding[0, :128], 0.0)


def test_stability_prefers_confident_masks():
    confident = np.full((4, 4), 5.0, np.float32)
    assert float(video.stability(confident)) == 1.0
    uncertain = np.zeros((4, 4), np.float32)
    assert float(video.stability(uncertain)) == 0.0


def test_select_stable_falls_back_to_the_best_mask():
    masks = np.zeros((1, 4, 8, 8), np.float32)
    masks[0, 2] = 4.0
    scores = np.array([[0.1, 0.2, 0.9, 0.3]], np.float32)
    tokens = np.arange(4 * 256, dtype="float32").reshape(1, 4, 256)
    chosen, token = video.select_stable(masks, scores, tokens)
    assert np.allclose(np.array(chosen)[0], 4.0)
    assert np.allclose(np.array(token), tokens[:, 0])


def test_select_best_takes_the_highest_score():
    masks = np.zeros((1, 4, 8, 8), np.float32)
    masks[0, 3] = 7.0
    scores = np.array([[0.9, 0.2, 0.3, 0.8]], np.float32)
    tokens = np.arange(4 * 256, dtype="float32").reshape(1, 4, 256)
    chosen, token = video.select_best(masks, scores, tokens)
    assert np.allclose(np.array(chosen)[0], 7.0)
    assert np.allclose(np.array(token), tokens[:, 3])


def test_build_memory_inputs_shapes():
    bundle = sam2_model.build_video(TINY)
    constants = video.build_constants(bundle)
    grid = np.zeros((1, GRID, GRID, 256), np.float32)
    features = video.Frame(None, grid, None, None)
    track = build_track([0], [8, 9])
    arguments = bundle, features, constants, track, 10, 12
    inputs = video.build_memory_inputs(*arguments)
    curr, curr_pos, memory, memory_pos, times, _, _, cos, sin = inputs
    assert curr.shape == (1, GRID * GRID, 256)
    assert curr_pos.shape == (1, GRID * GRID, 256)
    assert memory.shape[1] == 3 * GRID * GRID + 3 * video.POINTER_SPLIT
    assert memory.shape == memory_pos.shape
    assert times.shape[1] == memory.shape[1]
    assert cos.shape[1] == memory.shape[1]
    assert sin.shape == cos.shape
