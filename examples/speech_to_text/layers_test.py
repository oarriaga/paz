import numpy as np
from keras import ops

from examples.speech_to_text.layers import WhisperAudioFrontend
from examples.speech_to_text.layers import WhisperAttention
from examples.speech_to_text.layers import WhisperDecoderBlock
from examples.speech_to_text.layers import WhisperEncoderBlock
from examples.speech_to_text.layers import WhisperMelSpectrogram
from examples.speech_to_text.layers import WhisperPositionEmbedding
from examples.speech_to_text.layers import WhisperTokenAndPositionEmbedding
from examples.speech_to_text.layers import build_fixed_length_waveform
from examples.speech_to_text.layers import build_stft_waveform
from examples.speech_to_text.layers import build_decoder_self_attention_mask
from examples.speech_to_text.layers import compute_power_spectrogram
from examples.speech_to_text.layers import compute_stft_components
from examples.speech_to_text.layers import whisper_kernel_initializer
from examples.speech_to_text.weights import build_reference_whisper_audio_converter
from examples.speech_to_text.weights import build_whisper_frontend_waveform
from examples.speech_to_text.weights import build_whisper_frontend_waveform_batch
from examples.speech_to_text.weights import build_reference_whisper_model


def test_fixed_length_waveform_pads_short_waveform():
    waveform = ops.convert_to_tensor([[1.0, 2.0]], dtype="float32")
    padded_waveform = build_fixed_length_waveform(waveform, 5)
    expected = np.array([[1.0, 2.0, 0.0, 0.0, 0.0]], dtype="float32")
    np.testing.assert_array_equal(ops.convert_to_numpy(padded_waveform), expected)


def test_fixed_length_waveform_trims_long_waveform():
    waveform = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
    trimmed_waveform = build_fixed_length_waveform(waveform, 2)
    expected = np.array([[1.0, 2.0]], dtype="float32")
    np.testing.assert_array_equal(ops.convert_to_numpy(trimmed_waveform), expected)


def test_power_spectrogram_shape_matches_expected():
    waveform = ops.ones((1, 500), dtype="float32")
    waveform = build_stft_waveform(waveform, 400)
    real_part, imaginary_part = compute_stft_components(waveform, 400, 100)
    power_spectrogram = compute_power_spectrogram(real_part, imaginary_part)
    assert tuple(power_spectrogram.shape) == (1, 5, 201)


def test_mel_spectrogram_output_shape_matches_expected():
    layer = WhisperMelSpectrogram(
        num_mels=80,
        num_fft_bins=400,
        sampling_rate=100,
        dtype="float32",
    )
    inputs = ops.ones((1, 5, 201), dtype="float32")
    outputs = layer(inputs)
    assert tuple(outputs.shape) == (1, 5, 80)


def test_whisper_audio_frontend_matches_reference_example_values():
    layer = WhisperAudioFrontend(
        num_mels=80,
        num_fft_bins=400,
        stride=100,
        sampling_rate=100,
        max_audio_length=5,
        dtype="float32",
    )
    waveform = ops.ones((2,), dtype="float32")
    outputs = ops.convert_to_numpy(layer(waveform))
    expected = np.array(
        [1.1656, 1.0151, -0.8343, -0.8343, -0.8343], dtype="float32"
    )
    np.testing.assert_allclose(outputs[:, 0], expected, rtol=0.01, atol=0.01)


def test_whisper_audio_frontend_matches_reference():
    clean_layer = WhisperAudioFrontend(dtype="float32")
    reference_layer = build_reference_whisper_audio_converter(dtype="float32")
    waveform = build_whisper_frontend_waveform()
    clean_output = ops.convert_to_numpy(clean_layer(waveform))
    reference_output = reference_layer(ops.convert_to_numpy(waveform)).numpy()
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=2e-5)


def test_whisper_audio_frontend_matches_reference_batch():
    clean_layer = WhisperAudioFrontend(dtype="float32")
    reference_layer = build_reference_whisper_audio_converter(dtype="float32")
    waveform_batch = build_whisper_frontend_waveform_batch()
    clean_output = ops.convert_to_numpy(clean_layer(waveform_batch))
    reference_output = reference_layer(
        ops.convert_to_numpy(waveform_batch)
    ).numpy()
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=2e-5)


def test_encoder_position_embedding_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.ones((2, 3, 4), dtype="float32")
    reference_layer = reference_model.encoder_position_embedding
    clean_layer = WhisperPositionEmbedding(
        sequence_length=3,
        initializer=whisper_kernel_initializer(),
        trainable=False,
        name="encoder_position_embedding",
    )
    clean_layer(hidden)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(clean_layer(hidden))
    reference_output = ops.convert_to_numpy(reference_layer(hidden))
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)


def test_decoder_token_and_position_embedding_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    token_ids = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
    reference_layer = reference_model.decoder_embeddings
    clean_layer = WhisperTokenAndPositionEmbedding(
        vocabulary_size=10,
        sequence_length=6,
        embedding_dim=4,
        embeddings_initializer=whisper_kernel_initializer(),
        name="decoder_token_and_position_embedding",
    )
    clean_layer(token_ids)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(clean_layer(token_ids))
    reference_output = ops.convert_to_numpy(reference_layer(token_ids))
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)


def test_decoder_mask_matches_reference():
    decoder_sequence = ops.ones((1, 4, 4), dtype="float32")
    decoder_padding_mask = ops.convert_to_tensor([[1, 1, 1, 0]], dtype="int32")
    clean_mask = ops.convert_to_numpy(
        build_decoder_self_attention_mask(decoder_sequence, decoder_padding_mask)
    )
    expected_mask = np.array(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(clean_mask, expected_mask)


def test_self_attention_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer(
        "transformer_encoder_layer_0"
    )._self_attention_layer
    clean_layer = WhisperAttention(
        num_heads=2,
        key_dim=2,
        dropout=0.0,
        kernel_initializer=whisper_kernel_initializer(),
        bias_initializer="zeros",
        name="self_attention_layer",
    )
    clean_layer.build(hidden.shape, hidden.shape)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(clean_layer(hidden, hidden))
    reference_output = ops.convert_to_numpy(reference_layer(hidden, hidden))
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)


def test_cross_attention_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer(
        "transformer_decoder_layer_0"
    )._cross_attention_layer
    clean_layer = WhisperAttention(
        num_heads=2,
        key_dim=2,
        value_dim=2,
        dropout=0.0,
        kernel_initializer=whisper_kernel_initializer(),
        bias_initializer="zeros",
        name="cross_attention",
    )
    clean_layer.build(decoder_hidden.shape, encoder_hidden.shape)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(clean_layer(decoder_hidden, encoder_hidden))
    reference_output = ops.convert_to_numpy(
        reference_layer(decoder_hidden, encoder_hidden)
    )
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)


def test_encoder_block_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    reference_layer = reference_model.get_layer("transformer_encoder_layer_0")
    clean_layer = WhisperEncoderBlock(
        num_heads=2,
        intermediate_dim=8,
        dropout=0.0,
        name="transformer_encoder_layer_0",
    )
    clean_layer(hidden)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(clean_layer(hidden))
    reference_output = ops.convert_to_numpy(reference_layer(hidden))
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)


def test_decoder_block_matches_reference():
    reference_model = build_reference_whisper_model(10, 1, 2, 4, 8, 80, 0.0, 6, 6)
    decoder_hidden = ops.convert_to_tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]], dtype="float32"
    )
    encoder_hidden = ops.convert_to_tensor(
        [[[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]]], dtype="float32"
    )
    decoder_padding_mask = ops.convert_to_tensor([[1, 1]], dtype="int32")
    reference_layer = reference_model.get_layer("transformer_decoder_layer_0")
    clean_layer = WhisperDecoderBlock(
        num_heads=2,
        intermediate_dim=8,
        dropout=0.0,
        name="transformer_decoder_layer_0",
    )
    clean_layer(decoder_hidden, encoder_hidden, decoder_padding_mask)
    clean_layer.set_weights(reference_layer.get_weights())
    clean_output = ops.convert_to_numpy(
        clean_layer(decoder_hidden, encoder_hidden, decoder_padding_mask)
    )
    reference_output = ops.convert_to_numpy(
        reference_layer(
            decoder_hidden,
            encoder_hidden,
            decoder_padding_mask=decoder_padding_mask,
        )
    )
    np.testing.assert_allclose(clean_output, reference_output, rtol=1e-5, atol=1e-5)
