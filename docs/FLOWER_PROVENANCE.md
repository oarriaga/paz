# FLOWER VLA provenance

Inference-first port of FLOWER (Florence With Embodied Flow) to paz.

## Sources

- intuitive-robots/flower_vla_calvin
  commit acb32ee85719e51aa94139b184c64b1b29f11d33
  (`flower/models/flower.py::FLOWERVLA` is the class the released LIBERO
  checkpoints target; numerical source of truth.)
- intuitive-robots/flower_vla_pret
  commit 821185f1e16d9a09c3a9cf58b1caf0abac85afac
- Checkpoint: Hugging Face `mbreuss/flower_libero_object`
  revision 6353955e81eec05bdb084572ad204898cecd1950
  (`model.safetensors`, fp32, 4 000 241 052 bytes)
- General pretrained: `mbreuss/flower_vla_pret`
  revision ffd4134a2167a2be32bd7ad93455ed78dc2dcda2 (not converted yet)
- Backbone: `microsoft/Florence-2-large` (architecture reference only;
  all converted tensors come from the FLOWER checkpoint itself)

## Checkpoint architecture (from config.yaml + state dict)

- Florence-2-large truncated to encoder path only: DaViT vision tower,
  image projection 2048 -> 1024 with LayerNorm, learned 2D row/column
  position embeddings (50 x 50), visual temporal embedding, BART text
  encoder (12 layers, hidden 1024, ffn 4096, vocab 51289 + 1).
  The Florence text decoder and language-model head are not present in
  the checkpoint and are not executed.
- Flow DiT: 18 blocks, hidden 1024, 16 heads, causal self-attention with
  fused QKV, per-head RMS query/key norm, RoPE with theta 32, cross
  attention to Florence tokens, SwiGLU MLP (inner 2816), AdaLN-Zero
  bottleneck 1024 -> 256 -> 6144, per-action-space cross-attention
  modulation, weight-only RMS norms.
- Conditioning: sinusoidal timestep embedding + MLP, control-frequency
  embedding + MLP, learned action-space embedding table + MLP.
- Action spaces: eef_delta, joint_single, bimanual_nav; each with a
  two-layer action encoder and a linear decoder. LIBERO uses eef_delta,
  action dimension 7, chunk length 10.
- Sampling: uniform-time rectified flow, 4 Euler steps, integration from
  noise to actions.

## Preprocessing (exact, from checkpoint config.yaml val transforms)

- Static view: LIBERO `agentview_image`; wrist view:
  `robot0_eye_in_hand_image`.
- Resize to 112 x 112 (antialias), scale by 1/255, normalize with CLIP
  statistics mean (0.48145466, 0.4578275, 0.40821073) and
  std (0.26862954, 0.26130258, 0.27577711). No flipping.
- Language: Florence BART tokenizer, FLOWER "default" prompt style.

## Rollout convention (from flower/rollout/libero_rollout.py)

- Replan every 10 environment steps (multistep = act_seq_len = 10); the
  10-action chunk executes open loop between replans.
- Raw 7D delta end-effector actions passed directly to LIBERO
  `env.step`; gripper follows the robosuite convention (-1 open,
  +1 close). After `set_init_state`, 5 zero-action steps settle physics.

## Licenses

- FLOWER (intuitive-robots): MIT.
- Florence-2 (Microsoft): MIT.
- LIBERO: MIT.

## Converted weights

Hosted on the `oarriaga/altamira-data` GitHub release `v0.29` as sharded
assets with a reassembly manifest; `FLOWER(weights="pretrained")`
downloads, reassembles, and checksum-verifies them into the Keras cache.

- `florence2.weights.h5` sha256
  27d4197375f1c904c77fda85266e87f6a258d27d73d1a2342cfb6705a1c4d956
- `flower_dit.weights.h5` sha256
  7646f56063c62a85d3cc63a5df673cbd1775738b0971616ef727483bc1ec2888
- `tokenizer.json` sha256
  847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e

Parity results and known limitations are recorded in the pull request
body and test suite.
