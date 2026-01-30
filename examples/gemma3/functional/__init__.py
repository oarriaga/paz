from . import gemma3 as g3

apply_reversible_embedding = g3.apply_reversible_embedding
apply_reversible_projection = g3.apply_reversible_projection
apply_rotary_embedding = g3.apply_rotary_embedding
apply_tanh_soft_cap = g3.apply_tanh_soft_cap
build_cache = g3.build_cache
build_decoder_block = g3.build_decoder_block
build_gemma3_attention = g3.build_gemma3_attention
build_gemma3_backbone = g3.build_gemma3_backbone
build_gemma3_backbone_model = g3.build_gemma3_backbone_model
build_gemma3_causal_lm_model = g3.build_gemma3_causal_lm_model
build_reversible_embedding = g3.build_reversible_embedding
build_rms_norm = g3.build_rms_norm
build_vision_encoder = g3.build_vision_encoder
call_with_cache = g3.call_with_cache
compute_causal_mask = g3.compute_causal_mask
compute_num_vision_tokens_per_image = g3.compute_num_vision_tokens_per_image
interleave_embeddings = g3.interleave_embeddings
merge_padding_and_attention_mask = g3.merge_padding_and_attention_mask
