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

# Mapping from HuggingFace model IDs to (config, checkpoint) filenames.
HF_MODEL_ID_TO_FILENAMES = {
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


# ──────────────────────────────────────────────────────────────────────────────
# Device helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_best_available_device() -> str:
    """Return ``'cuda'``, ``'mps'``, or ``'cpu'`` depending on availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Core builders
# ──────────────────────────────────────────────────────────────────────────────

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
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
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


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hf_download(model_id: str) -> tuple[str, str]:
    """Download a SAM2.1 checkpoint from HuggingFace Hub.

    Parameters
    ----------
    model_id : str
        HuggingFace repository ID (see ``HF_MODEL_ID_TO_FILENAMES``).

    Returns
    -------
    tuple[str, str]
        ``(config_name, local_checkpoint_path)``
    """
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id: str, **kwargs):
    """Build a SAM2 model from a HuggingFace Hub model ID."""
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str | None) -> None:
    """Load a ``model`` state-dict from *ckpt_path* (if provided).

    Parameters
    ----------
    model : torch.nn.Module
    ckpt_path : str or None
        Path to a ``.pt`` file containing ``{"model": state_dict}``.
        When ``None`` the model is left with its initialised weights.

    Raises
    ------
    RuntimeError
        If the checkpoint has missing or unexpected keys.
    """
    if ckpt_path is None:
        return

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(sd)
    if missing:
        logging.error("Missing keys: %s", missing)
        raise RuntimeError("Checkpoint has missing keys.")
    if unexpected:
        logging.error("Unexpected keys: %s", unexpected)
        raise RuntimeError("Checkpoint has unexpected keys.")
    logging.info("Loaded checkpoint from %s", ckpt_path)
