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

Changes from the original
--------------------------
BUG FIXES
* find_components: replaced non-existent ``padding`` kwarg with the correct
  ``slice_pad`` and ``planar_pad`` — they are now separate arguments because
  the Z axis is measured in slices while Y/X are measured in voxels.
  ``--bbox_shift`` controls planar padding (Y/X); ``--slice_pad`` controls
  the axial padding (default 1).
* comp["z_mid"]: replaced with ``get_z_prompt_from_component(comp)`` since
  "z_mid" does not exist in the updated slicer.py.

NEW FEATURES
* ``--bbox_mode``   : gt | pet | unet | hybrid
* ``--pet_method``  : base41 | nestle | black | daisne
* ``--proposal_model``: path to Small3DUNet checkpoint
* ``--slice_pad``   : axial padding in slices (default 1)
* Rich per-patient JSON logs + ``inference_summary.csv`` in pred_save_dir

Usage (GT / oracle mode – unchanged)
-------------------------------------
python inference/infer_hecktor.py \\
    --checkpoint ./checkpoints/MedSAM2_latest.pt \\
    --cfg sam2/configs/sam2.1_hiera_tiny_hecktor.yaml \\
    --imgs_path /data/ethan/MedSAM2/hecktor_npz/val \\
    --pred_save_dir /data/ethan/MedSAM2/predictions/gt_oracle \\
    --bbox_mode gt --bbox_shift 5 --slice_pad 1

Usage (hybrid auto-prompting)
------------------------------
python inference/infer_hecktor.py \\
    ... \\
    --bbox_mode hybrid \\
    --pet_method base41 \\
    --proposal_model /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt \\
    --prob_threshold 0.25
"""

import argparse
import json
import csv
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
from slicer import find_components, scale_bbox_2d, get_z_prompt_from_component

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
    """Convert (D,H,W) uint8 CT and PET arrays to a normalized tensor.

    Output shape:
        (D, 3, S, S)

    Channel layout:
        [CT, PET, PET]
    """
    D = ct_imgs.shape[0]
    out = np.empty((D, 3, target_size, target_size), dtype=np.float32)

    for i in range(D):
        ct = np.array(
            Image.fromarray(ct_imgs[i]).resize((target_size, target_size)),
            dtype=np.float32,
        ) / 255.0

        pet = np.array(
            Image.fromarray(pet_imgs[i]).resize((target_size, target_size)),
            dtype=np.float32,
        ) / 255.0

        out[i, 0] = ct
        out[i, 1] = pet
        out[i, 2] = pet

    tensor = torch.from_numpy(out)

    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]

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
        for reverse in (False, True):  # forward, reverse
            state = predictor.init_state(img_tensor, video_h, video_w)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=key_z,
                obj_id=1,
                box=bbox_model,
            )

            for frame_idx, _, logits in predictor.propagate_in_video(
                state,
                reverse=reverse,
            ):
                _write(frame_idx, logits)

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
    log_dir: str | None,
    bbox_shift: int = 5,
    slice_pad: int = 1,
    bbox_mode: str = "gt",
    auto_prompter=None,
) -> tuple[str, float, dict[str, int]]:
    """Run MedSAM2 inference on a single HECKTOR NPZ volume.

    Parameters
    ----------
    npz_path : str
        Path to input NPZ file.

    predictor :
        SAM2 video predictor instance.

    pred_save_dir : str
        Directory where compressed segmentation NPZ files are saved.

    nifti_dir : str | None
        Optional output directory for NIfTI segmentations.

    overlay_dir : str | None
        Optional output directory for PNG overlay visualizations.

    log_dir : str | None
        Optional output directory for per-patient JSON logs.

    bbox_shift : int
        In-plane (Y/X) padding added around GT bounding boxes in voxels.

    slice_pad : int
        Axial (Z) padding added around GT components in slices.

    bbox_mode : str
        Bounding-box prompting mode:
            - "gt"     : oracle GT boxes
            - "pet"    : PET threshold proposals
            - "unet"   : proposal network
            - "hybrid" : combined PET + proposal model

    auto_prompter :
        AutoPrompter instance. Required when bbox_mode != "gt".

    Returns
    -------
    tuple[str, float, dict[str, int]]
        (
            patient_id,
            inference_duration_seconds,
            {
                "gtvp": int,
                "gtvn": int,
            }
        )
    """

    t0 = time.time()
    npz_name = basename(npz_path)
    patient_id = npz_name.replace(".npz", "")

    print(f"\n▶ {patient_id}")

    # ── Load ────────────────────────────────────────────────────────────
    data     = np.load(npz_path, allow_pickle=True)
    ct_imgs  = data["ct_imgs"]   # (D, H, W) uint8
    pet_imgs = data["pet_imgs"]  # (D, H, W) uint8
    gts      = data["gts"]       # (D, H, W) uint8  labels 0/1/2
    spacing  = data["spacing"]   # (3,) float64

    # ──────────────────────────────────────────────────────────────────
    # Load NPZ
    # ──────────────────────────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=True)

    ct_imgs = data["ct_imgs"]       # (D, H, W) uint8
    pet_imgs = data["pet_imgs"]     # (D, H, W) uint8
    gts = data["gts"]               # (D, H, W) uint8
    spacing = data["spacing"]       # (3,) float64

    # Optional field in newer NPZs
    pet_suv_max = (
        float(data["pet_suv_max"])
        if "pet_suv_max" in data
        else None
    )

    if pet_suv_max is None and bbox_mode in ("pet", "hybrid"):
        print(
            "  [WARN] pet_suv_max missing. "
            "Black/Daisne/Nestle methods will fall back to base41."
        )

    D, H, W = ct_imgs.shape

    segs_3d = np.zeros((D, H, W), dtype=np.uint8)
    orig_hw = (H, W)

    # ──────────────────────────────────────────────────────────────────
    # Build model input tensor
    # ──────────────────────────────────────────────────────────────────
    img_tensor = fuse_ct_pet_to_tensor(ct_imgs, pet_imgs, MODEL_IMG_SIZE).cuda()

    # ──────────────────────────────────────────────────────────────────
    # Get prompting components
    # ──────────────────────────────────────────────────────────────────
    if bbox_mode == "gt":
        components_per_label = find_components(
            mask=gts,
            slice_pad=slice_pad,
            planar_pad=bbox_shift,
            label_values=(LABEL_GTVp, LABEL_GTVn),
        )
    else:
        if auto_prompter is None:
            raise RuntimeError(
                f"bbox_mode='{bbox_mode}' requires an AutoPrompter "
                "instance. Provide --proposal_model or use --bbox_mode gt."
            )
        components_per_label = auto_prompter.get_proposals(
            ct_uint8  = ct_imgs,
            pet_uint8 = pet_imgs,
            suv_max   = pet_suv_max,
        )

    # ──────────────────────────────────────────────────────────────────
    # Propagation
    # ──────────────────────────────────────────────────────────────────
    n_gtvp = 0
    n_gtvn = 0

    if not components_per_label:
        print("  [WARN] No foreground components found, saving empty mask.")
    else:
        for label_id in sorted(components_per_label):
            components = components_per_label[label_id]
            label_name = "GTVp" if label_id == LABEL_GTVp else "GTVn"

            print(f"  {label_name}: {len(components)} component(s)")

            if label_id == LABEL_GTVp:
                n_gtvp = len(components)
            else:
                n_gtvn = len(components)

            for component in components:
            # GT slicer output and auto-prompter output use
                # slightly different schemas.
                if bbox_mode == "gt":
                    key_z, bbox_2d = get_z_prompt_from_component(component)
                else:
                    key_z = component["z_mid"]
                    bbox_2d = component["bbox_2d"]

                print(
                    f"    component={component['component_id']}  "
                    f"key_z={key_z}  "
                    f"bbox={bbox_2d.tolist()}  "
                    f"voxels={component['voxel_count']}"
                )

                propagate_one_component(
                    predictor=predictor,
                    img_tensor=img_tensor,
                    video_h=H,
                    video_w=W,
                    key_z=key_z,
                    bbox_orig=bbox_2d,
                    orig_hw=orig_hw,
                    label_id=label_id,
                    segs_3d=segs_3d,
                )

    duration = time.time() - t0

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
        # Stored as (z, y, x); SimpleITK expects (x, y, z)
        sitk_seg.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
        sitk.WriteImage(
            sitk_seg,
            join(nifti_dir, npz_name.replace(".npz", "_seg.nii.gz")),
        )

    # ── Optional PNG overlay ──────────────────────────────────────────────
    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        gtvp_comps = components_per_label.get(LABEL_GTVp, [])
        if gtvp_comps:
            if bbox_mode == "gt":
                key_z_vis = get_z_prompt_from_component(gtvp_comps[0])[0]
            else:
                key_z_vis = gtvp_comps[0]["z_mid"]
        else:
            key_z_vis = D // 2
        save_overlay(
            ct_imgs, segs_3d, key_slice=key_z_vis,
            save_path=join(overlay_dir, npz_name.replace(".npz", "_overlay.png")),
        )

    # ──────────────────────────────────────────────────────────────────
    # Optional JSON logging
    # ──────────────────────────────────────────────────────────────────
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        log_entry = {
            "patient_id": patient_id,
            "bbox_mode": bbox_mode,
            "pet_method": getattr(auto_prompter, "pet_method", "n/a"),
            "prob_threshold": getattr(auto_prompter, "prob_threshold", "n/a"),
            "slice_pad": slice_pad,
            "planar_pad_vox": bbox_shift,
            "pet_suv_max": pet_suv_max,
            "volume_shape_dhw": list(ct_imgs.shape),
            "gtvp_components": n_gtvp,
            "gtvn_components": n_gtvn,
            "proposals_total": n_gtvp + n_gtvn,
            "duration_seconds": round(duration, 2),
        }

        with open(join(log_dir, f"{patient_id}.json"), "w") as f:
            json.dump(log_entry, f, indent=2)

    print(f"  Done in {duration:.1f}s")

    return (patient_id, duration, {"gtvp": n_gtvp, "gtvn": n_gtvn})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    cfg_path  = os.path.abspath(args.cfg)
    predictor = build_sam2_video_predictor_npz(cfg_path, args.checkpoint)

    # ── Build auto-prompter if needed ─────────────────────────────────────
    auto_prompter = None
    if args.bbox_mode != "gt":
        from auto_prompting import AutoPrompter
        auto_prompter = AutoPrompter(
            method         = args.bbox_mode,
            model_path     = args.proposal_model,
            pet_method     = args.pet_method,
            device         = "cuda" if torch.cuda.is_available() else "cpu",
            prob_threshold = args.prob_threshold,
            slice_pad      = args.slice_pad,
            planar_pad     = args.bbox_shift,
        )
        print(f"Auto-prompter: {auto_prompter}")

    npz_files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {args.imgs_path}")
    print(f"Found {len(npz_files)} file(s) to process.")

    nifti_dir   = join(args.pred_save_dir, "nifti")    if args.save_nifti    else None
    overlay_dir = join(args.pred_save_dir, "overlays") if args.save_overlays else None
    log_dir     = join(args.pred_save_dir, "logs")

    timing: OrderedDict = OrderedDict()
    summary = []

    for npz_path in tqdm(npz_files, desc="inference"):
        patient, dur, counts = infer_one_npz(
            npz_path      = npz_path,
            predictor     = predictor,
            pred_save_dir = args.pred_save_dir,
            nifti_dir     = nifti_dir,
            overlay_dir   = overlay_dir,
            log_dir       = log_dir,
            bbox_shift    = args.bbox_shift,
            slice_pad     = args.slice_pad,
            bbox_mode     = args.bbox_mode,
            auto_prompter = auto_prompter,
        )
        timing[patient] = dur
        summary.append({
            "patient":    patient,
            "duration_s": f"{dur:.2f}",
            "n_gtvp":     counts["gtvp"],
            "n_gtvn":     counts["gtvn"],
        })

    # ── Timing CSV ────────────────────────────────────────────────────────
    timing_csv = join(args.pred_save_dir, "inference_timing.csv")
    with open(timing_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "duration_s"])
        for n, d in timing.items():
            w.writerow([n, f"{d:.2f}"])

    # ── Summary CSV (timing + component counts + run config) ─────────────
    summary_csv = join(args.pred_save_dir, "inference_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient", "duration_s", "n_gtvp", "n_gtvn",
        ])
        w.writeheader()
        w.writerows(summary)

    # ── Run config JSON (for easy result reproducibility) ────────────────
    config_json = join(args.pred_save_dir, "run_config.json")
    with open(config_json, "w") as f:
        json.dump(vars(args), f, indent=2)

    total = sum(timing.values())
    print(f"\nAll done.  Total: {total:.1f}s  ({total/len(npz_files):.1f}s/patient)")
    print(f"Summary CSV  : {summary_csv}")
    print(f"Per-patient  : {log_dir}/")
    print(f"Run config   : {config_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MedSAM2 HECKTOR inference (GTVp + GTVn, multi-component).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Original arguments ────────────────────────────────────────────────
    p.add_argument("--checkpoint", default="/data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt")
    p.add_argument("--cfg", default="sam2/configs/sam2.1_hiera_tiny_hecktor.yaml")
    p.add_argument("-i", "--imgs_path", default="/data/ethan/MedSAM2/hecktor_npz/val")
    p.add_argument("-o", "--pred_save_dir", default="/data/ethan/MedSAM2/predictions/val")
    p.add_argument("--bbox_shift", type=int, default=5,
                   help="Planar (Y/X) bounding-box padding in voxels.")
    p.add_argument("--save_nifti",    action="store_true")
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--num_workers",   type=int, default=1)

    # ── NEW: separate axial padding ───────────────────────────────────────
    p.add_argument("--slice_pad", type=int, default=1,
                   help="Axial (Z) bounding-box padding in slices. "
                        "Kept small (default 1) because one Z slice = one video frame.")

    # ── NEW: auto-prompting ───────────────────────────────────────────────
    p.add_argument("--bbox_mode", default="gt",
                   choices=["gt", "pet", "unet", "hybrid"],
                   help="Prompt source. 'gt'=oracle (dev only), others=auto.")
    p.add_argument("--pet_method", default="base41",
                   choices=["base41", "nestle", "black", "daisne"],
                   help="PET thresholding strategy for 'pet'/'hybrid' modes.")
    p.add_argument("--proposal_model", default=None,
                   help="Path to Small3DUNet .pt checkpoint "
                        "(required for 'unet'/'hybrid').")
    p.add_argument("--prob_threshold", type=float, default=0.25,
                   help="UNet probability threshold (recall-biased default 0.25).")

    main(p.parse_args())
