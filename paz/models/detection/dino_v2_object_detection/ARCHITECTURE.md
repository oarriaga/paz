# RF-DETR Architecture

Architecture of the RF-DETR object detector as implemented in this directory
(DINOv2 backbone + multi-scale projector + deformable Group-DETR decoder).

```mermaid
flowchart TB
    IMG["Input image<br/>(B, H, W, 3)<br/>ImageNet normalized<br/>res 384-560 by variant"]

    subgraph BB["Backbone (Backbone layer)"]
        direction TB
        DINO["DINOv2 ViT encoder<br/>windowed attention<br/>patch 14/16, num_windows<br/>taps blocks out_feature_indexes<br/>e.g. [1,4,7,10]"]
        PROJ["MultiScaleProjector<br/>ConvX + Bottleneck + LayerNorm<br/>to out_channels = 256<br/>pyramid scales P3/P4/P5/P6"]
        DINO --> PROJ
    end

    POS["Position encodings<br/>(poss_all)"]

    subgraph TR["Deformable Transformer (Group-DETR v3)"]
        direction TB
        SRC["Flatten multi-scale srcs<br/>+ masks + level embeds"]
        TWO["Two-stage proposals<br/>gen_encoder_output_proposals<br/>enc_out_class / enc_out_bbox"]
        QRY["Learnable queries<br/>refpoint_embed (4D)<br/>query_feat (256)<br/>num_queries x group_detr"]
        DEC["TransformerDecoder<br/>x dec_layers (2-3)"]
        DLAYER["Per layer:<br/>1. Self-attn (grouped in train)<br/>2. MSDeformAttn cross-attn<br/>3. FFN<br/>iterative refpoint refine"]
        SRC --> TWO --> DEC
        QRY --> DEC
        DEC --> DLAYER
    end

    subgraph HEAD["Prediction Heads"]
        direction TB
        CLS["class_embed (Dense)<br/>to logits"]
        BOX["bbox_embed (MLP x3)<br/>bbox reparam to boxes"]
        SEG["segmentation_head<br/>(optional) to masks"]
    end

    OUT["Outputs:<br/>pred_logits, pred_boxes<br/>pred_masks?, aux_outputs?, enc_outputs?"]

    TRAIN["Training:<br/>SetCriterion<br/>Hungarian matcher<br/>focal/varifocal + L1 + GIoU"]
    INFER["Inference:<br/>PostProcess<br/>top-K select (num_select)"]

    IMG --> BB
    BB --> SRC
    BB --> POS
    POS --> DEC
    DLAYER --> HEAD
    HEAD --> OUT
    OUT --> TRAIN
    OUT --> INFER
```

## Key components

- **Backbone** (`models/backbone/backbone.py`): a `DinoV2` windowed-attention
  ViT feeds a `MultiScaleProjector` that emits 256-channel pyramid features.
- **Transformer** (`models/transformer_decoder_head/transformer.py`): two-stage
  encoder proposals + a decoder stack where each layer does self-attention,
  multi-scale deformable cross-attention (`MSDeformAttn`), and an FFN, with
  iterative reference-point refinement.
- **Queries** (`models/lwdetr/lwdetr.py`): `num_queries x group_detr` learnable
  reference points + features; all groups used in training, only the first
  group at inference.
- **Heads**: `class_embed` (Dense) and `bbox_embed` (3-layer MLP with bbox
  reparameterization), plus an optional segmentation head.
- **Variants** (`config.py`): Nano/Small/Base/Medium/Large differ in
  resolution, patch size, window count, decoder layers, and tapped block
  indices.
