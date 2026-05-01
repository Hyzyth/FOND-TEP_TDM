"""
infer_hecktor.py
================
MedSAM2 inference for HECKTOR Task-1: GTVp (label 1) and GTVn (label 2)
segmentation from dual-modality CT + PET volumes.

The script:
  1. Loads preprocessed NPZ files (produced by data_preparation/prepare_hecktor_npz.py).
  2. Runs forward + reverse propagation from the key slice (largest tumour cross-section)
     using either a bounding-box or point prompt.
  3. Saves per-patient NPZ segmentation results and optionally NIfTI masks.

Usage
-----
python inference/infer_hecktor.py \
    --checkpoint /data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt \
    --cfg sam2/configs/sam2.1_hiera_t512.yaml \
    --imgs_path /data/ethan/MedSAM2/hecktor_npz/val \
    --pred_save_dir /data/ethan/MedSAM2/predictions/val \
    --save_nifti \
    --num_workers 1
"""

import argparse
import os
import random
import time
from collections import OrderedDict
from glob import glob
from os.path import basename, join

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

# Project-level imports – adjust sys.path if running from repo root.
from sam2.build_sam import build_sam2_video_predictor_npz

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# ImageNet normalisation constants (same as training).
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)

# Target image size used by the MedSAM2 backbone.
MODEL_IMG_SIZE = 512

# HECKTOR label ids.
LABEL_GTVp = 1
LABEL_GTVn = 2


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def fuse_ct_pet_to_tensor(
    ct_imgs: np.ndarray,
    pet_imgs: np.ndarray,
    target_size: int = MODEL_IMG_SIZE,
) -> torch.Tensor:
    """Convert (D, H, W) CT and PET uint8 arrays into a normalised (D, 3, S, S) tensor.

    Channel layout: [CT, PET, PET] – mirrors the training fusion strategy.

    Parameters
    ----------
    ct_imgs, pet_imgs : np.ndarray
        Uint8 arrays of shape (D, H, W) in [0, 255].
    target_size : int
        Spatial size to resize each slice to (square).

    Returns
    -------
    torch.Tensor  shape (D, 3, target_size, target_size) float32
    """
    D = ct_imgs.shape[0]
    out = np.zeros((D, 3, target_size, target_size), dtype=np.float32)

    for i in range(D):
        ct_pil  = Image.fromarray(ct_imgs[i]).convert("RGB").resize(
            (target_size, target_size)
        )
        pet_pil = Image.fromarray(pet_imgs[i]).convert("RGB").resize(
            (target_size, target_size)
        )

        ct_arr  = np.array(ct_pil).astype(np.float32) / 255.0    # (H, W, 3)
        pet_arr = np.array(pet_pil).astype(np.float32) / 255.0   # (H, W, 3)

        # Channel 0 = CT, channels 1-2 = PET.
        out[i, 0] = ct_arr[:, :, 0]
        out[i, 1] = pet_arr[:, :, 0]
        out[i, 2] = pet_arr[:, :, 0]

    tensor = torch.from_numpy(out)           # (D, 3, H, W)

    # Apply ImageNet normalisation.
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std  = torch.tensor(IMG_STD,  dtype=torch.float32)[:, None, None]
    tensor = (tensor - mean) / std

    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def mask2d_to_bbox(mask2d: np.ndarray, shift: int = 0) -> np.ndarray:
    """Compute axis-aligned bounding box from a 2-D binary mask.

    Parameters
    ----------
    mask2d : np.ndarray  (H, W) bool/uint8
    shift  : int  Additional margin (pixels) added on every side.

    Returns
    -------
    np.ndarray  [x_min, y_min, x_max, y_max]
    """
    ys, xs = np.where(mask2d > 0)
    H, W = mask2d.shape
    x_min = max(0,   int(xs.min()) - shift)
    x_max = min(W-1, int(xs.max()) + shift)
    y_min = max(0,   int(ys.min()) - shift)
    y_max = min(H-1, int(ys.max()) + shift)
    return np.array([x_min, y_min, x_max, y_max])


def find_key_slice(gts_label: np.ndarray) -> int:
    """Return the axial index with the largest foreground area for *gts_label*.

    Parameters
    ----------
    gts_label : np.ndarray  (D, H, W) binary mask for a single label.

    Returns
    -------
    int  Slice index (0-based).
    """
    areas = gts_label.sum(axis=(1, 2))
    return int(np.argmax(areas))


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def save_overlay(
    ct_imgs: np.ndarray,
    segs_3d: np.ndarray,
    key_slice: int,
    save_path: str,
    label_colors: dict = None,
) -> None:
    """Save a 3-panel overlay (25th pct / key slice / 75th pct) as PNG.

    Parameters
    ----------
    ct_imgs : np.ndarray  (D, H, W) uint8
    segs_3d : np.ndarray  (D, H, W) uint8 with labels 0/1/2
    key_slice : int
    save_path : str
    label_colors : dict  {label_id: (r, g, b)} in [0, 1] range
    """
    if label_colors is None:
        label_colors = {1: (1.0, 0.2, 0.2), 2: (0.2, 0.6, 1.0)}

    D = ct_imgs.shape[0]
    indices = [D // 4, key_slice, 3 * D // 4]
    titles  = ["25th pct", "Key slice", "75th pct"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, idx, title in zip(axes, indices, titles):
        ax.imshow(ct_imgs[idx], cmap="gray")
        for label_id, color in label_colors.items():
            mask = (segs_3d[idx] == label_id).astype(np.float32)
            if mask.sum() == 0:
                continue
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., :3] = color
            rgba[..., 3]  = mask * 0.5
            ax.imshow(rgba)
        ax.set_title(f"{title} (z={idx})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Per-file inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def infer_one_npz(
    npz_path: str,
    predictor,
    pred_save_dir: str,
    nifti_dir: str | None,
    overlay_dir: str | None,
    bbox_shift: int = 0,
    use_gt_prompt: bool = False,
) -> tuple[str, float]:
    """Run MedSAM2 inference on a single NPZ file and save results.

    Parameters
    ----------
    npz_path : str
        Path to the input NPZ file.
    predictor :
        Instantiated SAM2VideoPredictorNPZ model.
    pred_save_dir : str
        Directory to write the output NPZ.
    nifti_dir : str or None
        If not None, also save NIfTI segmentation masks here.
    overlay_dir : str or None
        If not None, save PNG overlay visualisations here.
    bbox_shift : int
        Extra margin (pixels) added around the bounding-box prompt.
    use_gt_prompt : bool
        If True, derive the prompt from ground-truth masks (oracle mode).

    Returns
    -------
    tuple[str, float]
        (npz basename, inference duration in seconds)
    """
    t0 = time.time()
    npz_name = basename(npz_path)
    print(f"\n▶ {npz_name}")

    # ── Load ───────────────────────────────────────────────────────────────
    data      = np.load(npz_path, allow_pickle=True)
    ct_imgs   = data["ct_imgs"]    # (D, H, W) uint8
    pet_imgs  = data["pet_imgs"]   # (D, H, W) uint8
    gts       = data["gts"]        # (D, H, W) uint8  – used for prompt / eval
    spacing   = data["spacing"]    # (3,) float64

    D, H, W = ct_imgs.shape
    segs_3d  = np.zeros((D, H, W), dtype=np.uint8)

    # ── Build model input tensor ───────────────────────────────────────────
    img_tensor = fuse_ct_pet_to_tensor(ct_imgs, pet_imgs, MODEL_IMG_SIZE).cuda()

    # ── Segment each label independently ──────────────────────────────────
    for label_id in [LABEL_GTVp, LABEL_GTVn]:
        gt_label = (gts == label_id).astype(np.uint8)
        if gt_label.sum() == 0:
            print(f"  Label {label_id}: absent – skipping.")
            continue

        key_z = find_key_slice(gt_label)
        print(f"  Label {label_id}: key slice = {key_z}")

        # Derive bounding-box prompt from GT (oracle) or simulate it.
        bbox = mask2d_to_bbox(gt_label[key_z], shift=bbox_shift)

        # ── Forward propagation (key → last) ───────────────────────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(img_tensor, H, W)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=key_z,
                obj_id=1,
                box=bbox,
            )
            for out_frame, _, out_logits in predictor.propagate_in_video(state):
                segs_3d[out_frame][out_logits[0].squeeze(0).cpu().numpy() > 0] = label_id
            predictor.reset_state(state)

        # ── Reverse propagation (key → first) ──────────────────────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(img_tensor, H, W)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=key_z,
                obj_id=1,
                box=bbox,
            )
            for out_frame, _, out_logits in predictor.propagate_in_video(
                state, reverse=True
            ):
                segs_3d[out_frame][out_logits[0].squeeze(0).cpu().numpy() > 0] = label_id
            predictor.reset_state(state)

    # ── Save output NPZ ────────────────────────────────────────────────────
    os.makedirs(pred_save_dir, exist_ok=True)
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs_3d,
        spacing=spacing,
    )

    # ── Optionally save NIfTI ──────────────────────────────────────────────
    if nifti_dir is not None:
        os.makedirs(nifti_dir, exist_ok=True)
        sitk_seg = sitk.GetImageFromArray(segs_3d)
        sitk_seg.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
        sitk.WriteImage(
            sitk_seg,
            join(nifti_dir, npz_name.replace(".npz", "_seg.nii.gz")),
        )

    # ── Optionally save overlay PNG ────────────────────────────────────────
    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        key_z_gtvp = find_key_slice((gts == LABEL_GTVp).astype(np.uint8))
        save_overlay(
            ct_imgs,
            segs_3d,
            key_slice=key_z_gtvp,
            save_path=join(overlay_dir, npz_name.replace(".npz", "_overlay.png")),
        )

    duration = time.time() - t0
    print(f"  Done in {duration:.1f}s")
    return npz_name, duration


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Load model, discover NPZ files, run inference, and write timing CSV."""

    # ── Resolve config path to absolute (Hydra requirement) ───────────────
    cfg_path = os.path.abspath(args.cfg)

    predictor = build_sam2_video_predictor_npz(cfg_path, args.checkpoint)

    # ── Discover input files ───────────────────────────────────────────────
    npz_files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {args.imgs_path}")

    print(f"Found {len(npz_files)} files to process.")

    nifti_dir   = join(args.pred_save_dir, "nifti") if args.save_nifti else None
    overlay_dir = join(args.pred_save_dir, "overlays") if args.save_overlays else None

    # ── Inference loop ─────────────────────────────────────────────────────
    timing: OrderedDict = OrderedDict()
    for npz_path in tqdm(npz_files, desc="inference"):
        name, dur = infer_one_npz(
            npz_path=npz_path,
            predictor=predictor,
            pred_save_dir=args.pred_save_dir,
            nifti_dir=nifti_dir,
            overlay_dir=overlay_dir,
            bbox_shift=args.bbox_shift,
        )
        timing[name] = dur

    # ── Write timing CSV ───────────────────────────────────────────────────
    import csv
    csv_path = join(args.pred_save_dir, "inference_timing.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "duration_s"])
        for name, dur in timing.items():
            writer.writerow([name, f"{dur:.2f}"])

    total = sum(timing.values())
    print(f"\nAll done.  Total: {total:.1f}s  |  CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedSAM2 inference for HECKTOR head-and-neck tumour segmentation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt",
        help="Path to the MedSAM2 model checkpoint.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="sam2/configs/sam2.1_hiera_t512.yaml",
        help="Relative or absolute path to the model YAML config.",
    )
    parser.add_argument(
        "-i", "--imgs_path",
        type=str,
        default="/data/ethan/MedSAM2/hecktor_npz/val",
        help="Directory containing input NPZ files.",
    )
    parser.add_argument(
        "-o", "--pred_save_dir",
        type=str,
        default="/data/ethan/MedSAM2/predictions/val",
        help="Directory where output NPZ files will be written.",
    )
    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=0,
        help="Extra pixel margin around bounding-box prompt (default: 0).",
    )
    parser.add_argument(
        "--save_nifti",
        action="store_true",
        help="Also save segmentation masks as NIfTI files.",
    )
    parser.add_argument(
        "--save_overlays",
        action="store_true",
        help="Save PNG overlay visualisations.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    main(parser.parse_args())
