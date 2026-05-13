# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
sam2/build_sam.py
=================
Factory functions for constructing SAM2 / MedSAM2 models.

For HECKTOR inference use ``build_sam2_video_predictor_npz``, which returns
a ``SAM2VideoPredictorNPZ`` configured for slice-by-slice 3-D propagation.
"""
import logging
import os

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Register OmegaConf custom resolvers used by SAM2 configs
if not OmegaConf.has_resolver("times"):
    OmegaConf.register_new_resolver(
        "times",
        lambda x, y: float(x) * float(y),
    )

if not OmegaConf.has_resolver("divide"):
    OmegaConf.register_new_resolver(
        "divide",
        lambda x, y: float(x) / float(y),
    )

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def get_best_available_device():
    """
    Get the best available device in the order: CUDA, MPS, CPU
    Returns: device string for torch.device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_sam2(
    config_file: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    hydra_overrides_extra: list | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
):
    """Build a SAM2Base model.

    Parameters
    ----------
    config_file : str
        Hydra config name (relative to the ``sam2/configs/`` directory).
    ckpt_path : str, optional
        Path to a ``.pt`` checkpoint to load.
    device : str, optional
        Target device; auto-detected when ``None``.
    mode : str
        ``'eval'`` (default) or ``'train'``.
    hydra_overrides_extra : list, optional
        Additional Hydra overrides.
    apply_postprocessing : bool
        Enable dynamic multi-mask stability postprocessing.
    """
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    config_dir = os.path.abspath(os.path.dirname(config_file))
    config_name = os.path.splitext(os.path.basename(config_file))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        cfg = compose(
            config_name=config_name,
            overrides=hydra_overrides_extra,
        )

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_npz(
    config_file: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    hydra_overrides_extra: list | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
):
    """Build a ``SAM2VideoPredictorNPZ`` for 3-D medical image inference.

    This is the recommended entry-point for HECKTOR inference.  CT/PET slices
    are treated as video frames; the predictor propagates a 2-D prompt (box or
    points on the key slice) forward and backward through the volume.

    Parameters
    ----------
    config_file : str
        Hydra config name or absolute path.
    ckpt_path : str, optional
        Path to a ``.pt`` checkpoint.  Defaults to ``None`` (random weights).
    device : str, optional
        Target device; auto-detected when ``None``.
    mode : str
        ``'eval'`` (default).
    hydra_overrides_extra : list, optional
        Additional Hydra overrides.
    apply_postprocessing : bool
        Enable hole-filling and multi-mask stability postprocessing.
    """
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra + [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]

    hydra_overrides.extend(hydra_overrides_extra)

    config_dir = os.path.abspath(os.path.dirname(config_file))
    config_name = os.path.splitext(os.path.basename(config_file))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        cfg = compose(
            config_name=config_name,
            overrides=hydra_overrides,
        )

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        # 1. Load the file WITHOUT trying to access ["model"] yet
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        
        # 2. Safely check if the key exists, otherwise assume it's the raw state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            sd = checkpoint["model"]
        else:
            sd = checkpoint
            
        # 3. Load the weights into the model
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        
        if missing_keys:
            logging.error(f"Missing keys: {missing_keys}")
            raise RuntimeError("Checkpoint has missing keys.")
        if unexpected_keys:
            logging.error(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Checkpoint has unexpected keys.")
            
        logging.info("Loaded checkpoint successfully")
