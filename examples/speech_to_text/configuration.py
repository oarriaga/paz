from examples.speech_to_text.model import WHISPER_MODELS_DIR

CONFIGS = {
    "whisper_tiny_en": {
        "vocabulary_size": 51864,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "ffn_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_en": {
        "vocabulary_size": 51864,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_en": {
        "vocabulary_size": 51864,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "ffn_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_en": {
        "vocabulary_size": 51864,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_tiny_multi": {
        "vocabulary_size": 51865,
        "num_layers": 4,
        "num_heads": 6,
        "hidden_dim": 384,
        "ffn_dim": 1536,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_base_multi": {
        "vocabulary_size": 51865,
        "num_layers": 6,
        "num_heads": 8,
        "hidden_dim": 512,
        "ffn_dim": 2048,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_small_multi": {
        "vocabulary_size": 51865,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "ffn_dim": 3072,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_medium_multi": {
        "vocabulary_size": 51865,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "ffn_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
    "whisper_large_multi_v2": {
        "vocabulary_size": 51865,
        "num_layers": 32,
        "num_heads": 20,
        "hidden_dim": 1280,
        "ffn_dim": 5120,
        "num_mels": 80,
        "dropout": 0.0,
        "max_encoder_sequence_length": 3000,
        "max_decoder_sequence_length": 448,
    },
}


def to_model_args(model_name, models_path=WHISPER_MODELS_DIR):
    config = CONFIGS[model_name]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    hidden_dim = config["hidden_dim"]
    ffn_dim = config["ffn_dim"]
    dropout = config["dropout"]
    num_mels = config["num_mels"]
    vocabulary_size = config["vocabulary_size"]
    encoder_length = config["max_encoder_sequence_length"]
    decoder_length = config["max_decoder_sequence_length"]
    kwargs = {"weights": model_name, "models_path": models_path}
    shared_args = (num_layers, num_heads, hidden_dim, ffn_dim)
    encoder_args = (num_mels,) + shared_args + (encoder_length, dropout)
    cross_cache_args = (num_layers, num_heads, hidden_dim)
    decoder_args = (vocabulary_size,) + shared_args + (decoder_length, dropout)
    return encoder_args, cross_cache_args, decoder_args, kwargs
