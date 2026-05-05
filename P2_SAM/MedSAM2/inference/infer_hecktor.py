"""
infer_hecktor.py
================
MedSAM2 inference for HECKTOR Task-1: GTVp (label 1) and GTVn (label 2)
segmentation from dual-modality CT + PET volumes.

GT label convention (single combined mask):
    0 – background
    1 – GTVp  (primary tumour, typically one component)
    2 – GTVn  (nodal tumour, may be 0, 1 or many components)

The script:
  1. Loads preprocessed NPZ files produced by prepare_hecktor_npz.py.
  2. Uses slicer.py to find all connected components per label and derives
     a per-component bounding-box prompt.
  3. Runs bidirectional propagation (forward + reverse) from the key slice
     of each component.
  4. Saves per-patient NPZ segmentation results and optionally NIfTI masks
     and PNG overlay visualisations.

Usage
-----
python inference/infer_hecktor.py \
    --checkpoint /data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt \
    --cfg sam2/configs/sam2.1_hiera_t512.yaml \
    --imgs_path /data/ethan/MedSAM2/hecktor_npz/val \
    --pred_save_dir /data/ethan/MedSAM2/predictions/val \
    --save_nifti \
    --save_overlays \
    --num_workers 1
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from glob import glob
from os.path import basename, dirname, abspath, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from tqdm import tqdm

# Ensure repo root is on sys.path when called from a sub-directory
_REPO_ROOT = dirname(dirname(abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor_npz
from slicer import find_components, scale_bbox_2d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD  = (0.229, 0.224, 0.225)
MODEL_IMG_SIZE = 512
LABEL_GTVp = 1
LABEL_GTVn = 2

torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def fuse_ct_pet_to_tensor(
    ct_imgs: np.ndarray,
    pet_imgs: np.ndarray,
    target_size: int = MODEL_IMG_SIZE,
) -> torch.Tensor:
    """Convert (D, H, W) uint8 CT and PET arrays to a (D, 3, S, S) tensor.

    Channel layout mirrors training: [CT, PET, PET].

    Parameters
    ----------
    ct_imgs, pet_imgs : (D, H, W) uint8 in [0, 255]
    target_size : int  square spatial size

    Returns
    -------
    torch.Tensor  (D, 3, target_size, target_size) float32, ImageNet-normalised
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
        ct_arr  = np.array(ct_pil,  dtype=np.float32) / 255.0
        pet_arr = np.array(pet_pil, dtype=np.float32) / 255.0
        out[i, 0] = ct_arr[:, :, 0]   # CT channel
        out[i, 1] = pet_arr[:, :, 0]  # PET channel
        out[i, 2] = pet_arr[:, :, 0]  # PET repeated

    tensor = torch.from_numpy(out)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std  = torch.tensor(IMG_STD,  dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_overlay(
    ct_imgs: np.ndarray,
    segs_3d: np.ndarray,
    key_slice: int,
    save_path: str,
) -> None:
    """Save a 3-panel CT overlay (25th pct / key slice / 75th pct) as PNG.
    Parameters
    ----------
    ct_imgs : np.ndarray  (D, H, W) uint8
    segs_3d : np.ndarray  (D, H, W) uint8 with labels 0/1/2
    key_slice : int
    save_path : str
    label_colors : dict  {label_id: (r, g, b)} in [0, 1] range
    """
    label_colors = {LABEL_GTVp: (1.0, 0.2, 0.2), LABEL_GTVn: (0.2, 0.6, 1.0)}
    D = ct_imgs.shape[0]
    indices = [max(0, D // 4), key_slice, min(D - 1, 3 * D // 4)]
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


# ---------------------------------------------------------------------------
# Per-component bidirectional propagation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def propagate_one_component(
    predictor,
    img_tensor: torch.Tensor,
    video_h: int,
    video_w: int,
    key_z: int,
    bbox_orig: np.ndarray,
    orig_hw: tuple,
    label_id: int,
    segs_3d: np.ndarray,
) -> None:
    """Run forward + reverse propagation for a single lesion component.

    Results are written into *segs_3d* in-place.  To avoid overwriting
    voxels already assigned to a different label (e.g. GTVp overwritten by
    GTVn), the write is guarded: a predicted voxel is only assigned
    *label_id* when it is currently background (0).  GTVp is processed
    first (label 1 < label 2), so GTVn can never clobber GTVp predictions.

    Parameters
    ----------
    predictor    : SAM2VideoPredictorNPZ
    img_tensor   : (D, 3, H_model, W_model) tensor on CUDA
    video_h, video_w : original slice dimensions
    key_z        : axial prompt slice index
    bbox_orig    : [x_min, y_min, x_max, y_max] in original image space
    orig_hw      : (H, W) of the original slices
    label_id     : int  (1 = GTVp, 2 = GTVn)
    segs_3d      : (D, H, W) uint8 array – modified in-place
    """
    # Scale bbox from original resolution to model resolution
    bbox_model = scale_bbox_2d(bbox_orig, orig_hw, MODEL_IMG_SIZE)

    def _write(frame_idx: int, logits: torch.Tensor) -> None:
        """Write predicted mask into segs_3d, never overwriting existing labels."""
        pred_mask = logits[0].squeeze(0).cpu().numpy() > 0   # (H, W) bool
        # Only fill voxels that are still background to avoid race conditions
        # when multiple labels share overlapping propagation paths.
        writeable = pred_mask & (segs_3d[frame_idx] == 0)
        segs_3d[frame_idx][writeable] = label_id

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # ── Forward propagation (key_z → last slice) ───────────────────
        state = predictor.init_state(img_tensor, video_h, video_w)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_z,
            obj_id=1,
            box=bbox_model,
        )
        for out_frame, _, out_logits in predictor.propagate_in_video(state):
            _write(out_frame, out_logits)
        predictor.reset_state(state)

        # ── Reverse propagation (key_z → first slice) ──────────────────
        state = predictor.init_state(img_tensor, video_h, video_w)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=key_z,
            obj_id=1,
            box=bbox_model,
        )
        for out_frame, _, out_logits in predictor.propagate_in_video(
            state, reverse=True
        ):
            _write(out_frame, out_logits)
        predictor.reset_state(state)


# ---------------------------------------------------------------------------
# Per-file inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_one_npz(
    npz_path: str,
    predictor,
    pred_save_dir: str,
    nifti_dir: str | None,
    overlay_dir: str | None,
    bbox_shift: int = 0,
) -> tuple[str, float]:
    """Run MedSAM2 inference on one HECKTOR NPZ file and save results.

    Parameters
    ----------
    npz_path      : str   path to input NPZ
    predictor     : SAM2VideoPredictorNPZ
    pred_save_dir : str   directory for output NPZ
    nifti_dir     : str or None  optional NIfTI output directory
    overlay_dir   : str or None  optional PNG overlay directory
    bbox_shift    : int   extra margin (pixels) added around bounding-box prompts

    Returns
    -------
    (basename, duration_seconds)
    """
    t0 = time.time()
    npz_name = basename(npz_path)
    print(f"\n▶ {npz_name}")

    # ── Load ────────────────────────────────────────────────────────────
    data     = np.load(npz_path, allow_pickle=True)
    ct_imgs  = data["ct_imgs"]   # (D, H, W) uint8
    pet_imgs = data["pet_imgs"]  # (D, H, W) uint8
    gts      = data["gts"]       # (D, H, W) uint8  labels 0/1/2
    spacing  = data["spacing"]   # (3,) float64

    D, H, W = ct_imgs.shape
    segs_3d  = np.zeros((D, H, W), dtype=np.uint8)

    # ── Build model input tensor ─────────────────────────────────────────
    img_tensor = fuse_ct_pet_to_tensor(ct_imgs, pet_imgs, MODEL_IMG_SIZE).cuda()
    orig_hw = (H, W)

    # ── Use slicer.py to find per-label connected components + prompts ───
    # Process labels in ascending order so GTVp (1) is written before
    # GTVn (2); the guarded write in propagate_one_component then ensures
    # GTVn can never overwrite GTVp voxels.
    components_per_label = find_components(
        mask        = gts,
        padding     = bbox_shift,
        label_values= (LABEL_GTVp, LABEL_GTVn),
    )

    if not components_per_label:
        print(f"  [WARN] {npz_name}: no foreground labels found – saving empty mask.")
    else:
        for label_id in sorted(components_per_label):          # 1 before 2
            components = components_per_label[label_id]
            label_name = "GTVp" if label_id == LABEL_GTVp else "GTVn"
            print(f"  {label_name}: {len(components)} component(s)")

            for comp in components:
                key_z    = comp["z_mid"]
                bbox_2d  = comp["bbox_2d"]   # [x_min, y_min, x_max, y_max]
                print(
                    f"    component {comp['component_id']}: "
                    f"key_z={key_z}, bbox={bbox_2d.tolist()}, "
                    f"voxels={comp['voxel_count']}"
                )
                propagate_one_component(
                    predictor = predictor,
                    img_tensor= img_tensor,
                    video_h   = H,
                    video_w   = W,
                    key_z     = key_z,
                    bbox_orig = bbox_2d,
                    orig_hw   = orig_hw,
                    label_id  = label_id,
                    segs_3d   = segs_3d,
                )

    # ── Save output NPZ ─────────────────────────────────────────────────
    os.makedirs(pred_save_dir, exist_ok=True)
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs    = segs_3d,
        spacing = spacing,
    )

    # ── Optionally save NIfTI ────────────────────────────────────────────
    if nifti_dir is not None:
        os.makedirs(nifti_dir, exist_ok=True)
        sitk_seg = sitk.GetImageFromArray(segs_3d)
        # spacing stored as (z, y, x); SimpleITK expects (x, y, z)
        sitk_seg.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
        sitk.WriteImage(
            sitk_seg,
            join(nifti_dir, npz_name.replace(".npz", "_seg.nii.gz")),
        )

    # ── Optionally save overlay PNG ──────────────────────────────────────
    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        # Use GTVp key slice for the visualisation anchor
        gtvp_comps = components_per_label.get(LABEL_GTVp, [])
        key_z_vis = gtvp_comps[0]["z_mid"] if gtvp_comps else D // 2
        save_overlay(
            ct_imgs,
            segs_3d,
            key_slice=key_z_vis,
            save_path=join(overlay_dir, npz_name.replace(".npz", "_overlay.png")),
        )

    duration = time.time() - t0
    print(f"  Done in {duration:.1f}s")
    return npz_name, duration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg_path  = os.path.abspath(args.cfg)
    predictor = build_sam2_video_predictor_npz(cfg_path, args.checkpoint)

    npz_files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {args.imgs_path}")
    print(f"Found {len(npz_files)} file(s) to process.")

    nifti_dir   = join(args.pred_save_dir, "nifti")    if args.save_nifti    else None
    overlay_dir = join(args.pred_save_dir, "overlays") if args.save_overlays else None

    timing: OrderedDict = OrderedDict()
    for npz_path in tqdm(npz_files, desc="inference"):
        name, dur = infer_one_npz(
            npz_path      = npz_path,
            predictor     = predictor,
            pred_save_dir = args.pred_save_dir,
            nifti_dir     = nifti_dir,
            overlay_dir   = overlay_dir,
            bbox_shift    = args.bbox_shift,
        )
        timing[name] = dur

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
        description="MedSAM2 HECKTOR inference (GTVp + GTVn, multi-component)."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt",
    )
    parser.add_argument(
        "--cfg", type=str,
        default="sam2/configs/sam2.1_hiera_t512.yaml",
    )
    parser.add_argument(
        "-i", "--imgs_path", type=str,
        default="/data/ethan/MedSAM2/hecktor_npz/val",
    )
    parser.add_argument(
        "-o", "--pred_save_dir", type=str,
        default="/data/ethan/MedSAM2/predictions/val",
    )
    parser.add_argument(
        "--bbox_shift", type=int, default=5,
        help="Extra voxel margin around bounding-box prompts (default: 5).",
    )
    parser.add_argument("--save_nifti",    action="store_true")
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--num_workers",   type=int, default=1)
    main(parser.parse_args())
