import torch
from functools import partial
from torch.nn import functional as F

from sam_modeling_wave import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
    WaveEncoder,
)


# =========================
# SAM ViT-H builder
# =========================
def build_sam_vit_h(args):
    """
    Build SAM ViT-H variant.

    Args:
        args: configuration object containing image size,
              checkpoint path, and encoder adapter flag.
    """
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


build_sam = build_sam_vit_h


# =========================
# SAM ViT-L builder
# =========================
def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


# =========================
# SAM ViT-B builder
# =========================
def build_sam_vit_b(args):
    """
    Build SAM ViT-B variant.
    """
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


# =========================
# Dual-wave SAM builder
# =========================
def build_sam_dual_wave(args):
    """
    Build SAM variant with wavelet-based encoder.
    """
    return _build_sam_dual_wave(
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        wavelet=args.wavelet,
    )


# Model registry
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "dual_wave": build_sam_dual_wave,
}


# ============================================================
# Dual-wave SAM construction (wavelet-based encoder variant)
# ============================================================
def _build_sam_dual_wave(image_size, checkpoint, wavelet):
    """
    Build SAM with WaveEncoder backbone.

    Args:
        image_size: input resolution
        checkpoint: pretrained weight path
        wavelet: wavelet type used in encoder
    """

    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=WaveEncoder(wavelet=wavelet),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        # FIX: explicitly pass img_size so Sam.postprocess_masks works
        # with WaveEncoder (which has no .img_size attribute)
        img_size=image_size,
    )

    # Load checkpoint if provided
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        try:
            if "model" in state_dict.keys():
                sam.load_state_dict(state_dict["model"], strict=False)
            else:
                sam.load_state_dict(state_dict)
        except Exception:
            print("*******interpolate fallback triggered")
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)

        print(f"*******loaded {checkpoint}")

    return sam


# ============================================================
# Standard SAM builder (ViT backbone)
# ============================================================
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    encoder_adapter,
):
    """
    Construct standard SAM model with ViT encoder.

    Args:
        encoder_embed_dim: embedding dimension of encoder
        encoder_depth: transformer depth
        encoder_num_heads: number of attention heads
        encoder_global_attn_indexes: global attention layers
        image_size: input resolution
        checkpoint: pretrained weights path
        encoder_adapter: enable adapter modules
    """
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            adapter_train=encoder_adapter,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        img_size=image_size,
    )

    # Load pretrained weights
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        try:
            if "model" in state_dict.keys():
                print(encoder_adapter)
                sam.load_state_dict(state_dict["model"], strict=False)
            else:
                if image_size == 1024 and encoder_adapter is True:
                    sam.load_state_dict(state_dict, strict=False)
                else:
                    sam.load_state_dict(state_dict)

        except Exception:
            print("*******interpolate fallback triggered")
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)

        print(f"*******loaded {checkpoint}")

    return sam


# ============================================================
# Weight interpolation utility for mismatched checkpoints
# ============================================================
def load_from(sam, state_dicts, image_size, vit_patch_size):
    """
    Adapt pretrained weights to different image sizes or architectures.

    Handles:
    - Positional embedding resizing
    - Relative position embedding adjustment
    - Filtering incompatible keys
    """

    sam_dict = sam.state_dict()

    # Exclude task-specific heads
    except_keys = [
        "mask_tokens",
        "output_hypernetworks_mlps",
        "iou_prediction_head",
    ]

    new_state_dict = {
        k: v
        for k, v in state_dicts.items()
        if k in sam_dict.keys()
        and except_keys[0] not in k
        and except_keys[1] not in k
        and except_keys[2] not in k
    }

    # Only resize positional embeddings for ViT encoders
    if "image_encoder.pos_embed" not in new_state_dict:
        sam_dict.update(new_state_dict)
        return sam_dict

    pos_embed = new_state_dict["image_encoder.pos_embed"]
    token_size = int(image_size // vit_patch_size)

    # Resize positional embedding if resolution mismatch occurs
    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            (token_size, token_size),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)

        new_state_dict["image_encoder.pos_embed"] = pos_embed

        rel_pos_keys = [k for k in sam_dict.keys() if "rel_pos" in k]

        global_rel_pos_keys = [
            k for k in rel_pos_keys
            if any(idx in k for idx in ["2", "5", "7", "8", "11", "13", "15", "23", "31"])
        ]

        for k in global_rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]

            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)

            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(
                    rel_pos_params,
                    (h_check, w_check),
                    mode="bilinear",
                    align_corners=False,
                )

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict
