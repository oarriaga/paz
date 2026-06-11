# DINOv2 Architecture

Architecture of the DINOv2 Vision Transformer as implemented in this directory
(functional Keras `DinoVisionTransformer` in `models/vision_transformer.py`
with layer builders under `layers/`).

```mermaid
flowchart TB
    IMG["Input image<br/>(B, img_size, img_size, 3)<br/>img_size default 518"]

    subgraph EMB["Patch + Token embedding"]
        direction TB
        PATCH["build_patch_embedding<br/>Conv2D kernel=stride=patch_size<br/>then flatten to (N, embed_dim)<br/>N = (img_size/patch_size)^2"]
        CLS["prepend CLS token<br/>(RandomNormal)"]
        POS["add positional embedding<br/>(TruncatedNormal, N+1)"]
        REG["insert register tokens<br/>(optional, num_register_tokens)"]
        PATCH --> CLS --> POS --> REG
    end

    subgraph BLK["Transformer block x depth"]
        direction TB
        N1["LayerNorm (norm1)"]
        ATT["Multi-Head Self-Attention<br/>qkv Dense(3*dim) -> split q,k,v<br/>scaled dot-product (head_dim^-0.5)<br/>softmax -> attn_drop -> proj"]
        LS1["LayerScale (ls1)"]
        DP1["DropPath (drop_path1)"]
        ADD1["Add residual"]
        N2["LayerNorm (norm2)"]
        FFN["FFN<br/>MLP (fc1 -> GELU -> fc2)<br/>or SwiGLU (giant)"]
        LS2["LayerScale (ls2)"]
        DP2["DropPath (drop_path2)"]
        ADD2["Add residual"]
        N1 --> ATT --> LS1 --> DP1 --> ADD1
        ADD1 --> N2 --> FFN --> LS2 --> DP2 --> ADD2
    end

    NORM["Final LayerNorm (norm)"]
    OUT["take_first_token<br/>CLS output (B, embed_dim)"]

    IMG --> EMB
    EMB --> N1
    ADD2 --> NORM
    NORM --> OUT
```

## Block residual flow

Each block applies two pre-norm residual branches (DINOv2 / ViT style):

```
x = x + DropPath(LayerScale(Attention(LayerNorm(x))))
x = x + DropPath(LayerScale(FFN(LayerNorm(x))))
```

## Variants

| Builder      | embed_dim | depth | heads | FFN    |
| ------------ | --------- | ----- | ----- | ------ |
| `vit_small`  | 384       | 12    | 6     | MLP    |
| `vit_base`   | 768       | 12    | 12    | MLP    |
| `vit_large`  | 1024      | 24    | 16    | MLP    |
| `vit_giant2` | 1536      | 40    | 24    | SwiGLU |

Shared defaults: `patch_size=14`, `img_size=518`, `mlp_ratio=4.0`,
`init_values=1e-5`, qkv/proj/ffn bias enabled, `drop_path_rate=0.0`.

## Key components

- **Patch embedding** (`layers/patch_embed.py`): a single `Conv2D` with
  kernel and stride equal to `patch_size` projects non-overlapping patches,
  then reshapes to a token sequence.
- **Token stream** (`models/vision_transformer.py`): prepends a learnable
  CLS token, adds learnable positional embeddings, and optionally inserts
  register tokens after the CLS token.
- **Attention** (`layers/attention.py`): fused `qkv` Dense projection, head
  split, scaled dot-product attention with softmax and dropout, output
  projection.
- **FFN** (`layers/mlp.py`, `layers/swiglu_ffn.py`): standard GELU MLP for
  small/base/large; fused SwiGLU for the giant variant.
- **LayerScale** (`layers/layer_scale.py`): per-channel learnable scaling via
  `EinsumDense`, applied to each residual branch when `init_values` is set.
- **DropPath** (`layers/drop_path.py`): stochastic depth on residual branches
  (identity when the rate is 0).
- **Head**: final `LayerNorm` followed by selecting the CLS token as the
  model output.

## Attention internals

```mermaid
flowchart LR
    X["tokens (B, N+1, dim)"] --> QKV["Dense qkv (3*dim)"]
    QKV --> SPLIT["reshape + split<br/>q, k, v per head"]
    SPLIT --> SCORE["q . k^T * head_dim^-0.5"]
    SCORE --> SM["softmax + attn_drop"]
    SM --> AV["scores . v"]
    AV --> MERGE["merge heads + flatten"]
    MERGE --> PROJ["Dense proj + drop"]
    PROJ --> Y["output (B, N+1, dim)"]
```
