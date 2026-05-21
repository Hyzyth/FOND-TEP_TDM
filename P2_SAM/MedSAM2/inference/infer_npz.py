"""
infer_npz.py
=============
General MedSAM2 inference for any NPZ dataset (GTVp label=1, GTVn label=2).

Drop-in generalization of infer_hecktor.py. Accepts any directory of NPZ files
prepared by prepare_hecktor_npz.py or prepare_temporal_npz.py.

Usage (GT oracle):
    python inference/infer_npz.py \\
        --checkpoint ./checkpoints/MedSAM2_latest.pt \\
        --cfg sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml \\
        --imgs_path /data/ethan/MedSAM2/temporal_npz \\
        --pred_save_dir /data/ethan/MedSAM2/predictions/temporal_gt \\
        --bbox_mode gt

Usage (auto-prompting):
    python inference/infer_npz.py \\
        ... \\
        --bbox_mode hybrid \\
        --proposal_model /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from glob import glob
from os.path import abspath, basename, dirname, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = dirname(dirname(abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor_npz
from slicer import find_components, scale_bbox_2d, get_z_prompt_from_component

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


def fuse_ct_pet_to_tensor(ct_imgs, pet_imgs, target_size=MODEL_IMG_SIZE):
    """(D,H,W) uint8 CT + PET → normalised (D,3,S,S) tensor. Channel: [CT, PET, PET]."""
    D = ct_imgs.shape[0]
    out = np.empty((D, 3, target_size, target_size), dtype=np.float32)
    for i in range(D):
        ct  = np.array(Image.fromarray(ct_imgs[i]).resize((target_size, target_size)),
                       dtype=np.float32) / 255.0
        pet = np.array(Image.fromarray(pet_imgs[i]).resize((target_size, target_size)),
                       dtype=np.float32) / 255.0
        out[i, 0] = ct
        out[i, 1] = pet
        out[i, 2] = pet
    tensor = torch.from_numpy(out)
    mean = torch.tensor(IMG_MEAN)[:, None, None]
    std  = torch.tensor(IMG_STD)[:, None, None]
    return (tensor - mean) / std


def save_overlay(ct_imgs, segs_3d, key_slice, save_path):
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


@torch.inference_mode()
def propagate_one_component(predictor, img_tensor, video_h, video_w,
                             key_z, bbox_orig, orig_hw, label_id, segs_3d):
    bbox_model = scale_bbox_2d(bbox_orig, orig_hw, MODEL_IMG_SIZE)

    def _write(frame_idx, logits):
        pred_mask = logits[0].squeeze(0).cpu().numpy() > 0
        writeable  = pred_mask & (segs_3d[frame_idx] == 0)
        segs_3d[frame_idx][writeable] = label_id

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for reverse in (False, True):
            state = predictor.init_state(img_tensor, video_h, video_w)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=key_z, obj_id=1, box=bbox_model)
            for frame_idx, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                _write(frame_idx, logits)
            predictor.reset_state(state)


@torch.inference_mode()
def infer_one_npz(npz_path, predictor, pred_save_dir,
                  nifti_dir, overlay_dir, log_dir,
                  bbox_shift: int = 5, slice_pad: int = 1, bbox_mode: str = "gt", auto_prompter=None):
    t0 = time.time()
    npz_name   = basename(npz_path)
    case_id    = npz_name.replace(".npz", "")
    print(f"\n▶ {case_id}")

    data     = np.load(npz_path, allow_pickle=True)
    ct_imgs  = data["ct_imgs"]
    pet_imgs = data["pet_imgs"]
    gts      = data["gts"]
    spacing  = data["spacing"]
    pet_suv_max = float(data["pet_suv_max"]) if "pet_suv_max" in data else None

    if pet_suv_max is None and bbox_mode in ("pet", "hybrid"):
        print(
            "  [WARN] pet_suv_max missing. "
            "Black/Daisne/Nestle methods will fall back to base41."
        )

    D, H, W = ct_imgs.shape
    segs_3d = np.zeros((D, H, W), dtype=np.uint8)
    orig_hw = (H, W)

    img_tensor = fuse_ct_pet_to_tensor(ct_imgs, pet_imgs, MODEL_IMG_SIZE).cuda()

    if bbox_mode == "gt":
        components_per_label = find_components(
            mask=gts, slice_pad=slice_pad, planar_pad=bbox_shift,
            label_values=(LABEL_GTVp, LABEL_GTVn),
        )
    else:
        if auto_prompter is None:
            raise RuntimeError(f"bbox_mode='{bbox_mode}' requires an AutoPrompter.")
        components_per_label = auto_prompter.get_proposals(ct_uint8  = ct_imgs, pet_uint8 = pet_imgs, 
                                                           suv_max   = pet_suv_max)

    n_gtvp = n_gtvn = 0

    if not components_per_label:
        print("  [WARN] No foreground components — saving empty mask.")
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
                if bbox_mode == "gt":
                    key_z, bbox_2d = get_z_prompt_from_component(component)
                else:
                    key_z  = component["z_mid"]
                    bbox_2d = component["bbox_2d"]

                print(f"    comp={component['component_id']}  key_z={key_z}  "
                      f"bbox={bbox_2d.tolist()}  voxels={component['voxel_count']}")
                
                propagate_one_component(
                    predictor=predictor, img_tensor=img_tensor, video_h=H, video_w=W, key_z=key_z, 
                    bbox_orig=bbox_2d, orig_hw=orig_hw, label_id=label_id, segs_3d=segs_3d,
                )

    duration = time.time() - t0

    os.makedirs(pred_save_dir, exist_ok=True)
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3d, spacing=spacing)

    if nifti_dir is not None:
        os.makedirs(nifti_dir, exist_ok=True)
        sitk_seg = sitk.GetImageFromArray(segs_3d)
        sitk_seg.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
        sitk.WriteImage(sitk_seg,
                        join(nifti_dir, npz_name.replace(".npz", "_seg.nii.gz")))

    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        gtvp_comps = components_per_label.get(LABEL_GTVp, []) if components_per_label else []

        key_z_vis = (get_z_prompt_from_component(gtvp_comps[0])[0]
                     if gtvp_comps and bbox_mode == "gt"
                     else (gtvp_comps[0]["z_mid"] if gtvp_comps else D // 2))
        
        save_overlay(ct_imgs, segs_3d, key_slice=key_z_vis,
                     save_path=join(overlay_dir, npz_name.replace(".npz", "_overlay.png")))

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(join(log_dir, f"{case_id}.json"), "w") as f:
            json.dump({
                "case_id": case_id, "bbox_mode": bbox_mode,
                "pet_method": getattr(auto_prompter, "pet_method", "n/a"),
                "prob_threshold": getattr(auto_prompter, "prob_threshold", "n/a"),
                "slice_pad": slice_pad, "planar_pad_vox": bbox_shift,
                "pet_suv_max": pet_suv_max,
                "volume_shape_dhw": list(ct_imgs.shape),
                "gtvp_components": n_gtvp, "gtvn_components": n_gtvn, "proposals_total": n_gtvp + n_gtvn,
                "duration_seconds": round(duration, 2),
            }, f, indent=2)

    print(f"  Done in {duration:.1f}s")
    return case_id, duration, {"gtvp": n_gtvp, "gtvn": n_gtvn}


def main(args):
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)

    auto_prompter = None
    if args.bbox_mode != "gt":
        from auto_prompting import AutoPrompter
        auto_prompter = AutoPrompter(
            method=args.bbox_mode, model_path=args.proposal_model,
            pet_method=args.pet_method, device="cuda" if torch.cuda.is_available() else "cpu",
            prob_threshold=args.prob_threshold,
            slice_pad=args.slice_pad, planar_pad=args.bbox_shift,
        )
        print(f"Auto-prompter: {auto_prompter}")

    npz_files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    # Exclude manifest.json-adjacent metadata NPZs if any
    npz_files = [f for f in npz_files if not basename(f).startswith("_")]
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found under {args.imgs_path}")
    print(f"Found {len(npz_files)} file(s) to process.")

    nifti_dir   = join(args.pred_save_dir, "nifti")    if args.save_nifti    else None
    overlay_dir = join(args.pred_save_dir, "overlays") if args.save_overlays else None
    log_dir     = join(args.pred_save_dir, "logs")

    timing  = OrderedDict()
    summary = []

    for npz_path in tqdm(npz_files, desc="inference"):
        patient, dur, counts = infer_one_npz(
            npz_path=npz_path, predictor=predictor,
            pred_save_dir=args.pred_save_dir,
            nifti_dir=nifti_dir, overlay_dir=overlay_dir, log_dir=log_dir,
            bbox_shift=args.bbox_shift, slice_pad=args.slice_pad,
            bbox_mode=args.bbox_mode, auto_prompter=auto_prompter,
        )
        timing[patient] = dur
        summary.append({"patient": patient, "duration_s": f"{dur:.2f}",
                        "n_gtvp": counts["gtvp"], "n_gtvn": counts["gtvn"]})

    # ── Timing CSV ────────────────────────────────────────────────────────
    with open(join(args.pred_save_dir, "inference_timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "duration_s"])
        for n, d in timing.items():
            w.writerow([n, f"{d:.2f}"])

    # ── Summary CSV (timing + component counts + run config) ─────────────
    with open(join(args.pred_save_dir, "inference_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "duration_s", "n_gtvp", "n_gtvn"])
        w.writeheader(); w.writerows(summary)

    # ── Run config JSON (for easy result reproducibility) ────────────────
    with open(join(args.pred_save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    total = sum(timing.values())
    n = len(npz_files)
    print(f"\nAll done. Total: {total:.1f}s  ({total/n:.1f}s/case)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="General MedSAM2 NPZ inference (GTVp+GTVn).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="./checkpoints/MedSAM2_latest.pt")
    p.add_argument("--cfg",        default="sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml")
    p.add_argument("-i", "--imgs_path",     default="/data/ethan/MedSAM2/hecktor_npz/val")
    p.add_argument("-o", "--pred_save_dir", default="/data/ethan/MedSAM2/predictions/val")
    p.add_argument("--bbox_shift",  type=int,   default=5,
                   help="Planar (Y/X) bounding-box padding in voxels.")
    p.add_argument("--slice_pad",   type=int,   default=1,
                   help="Axial (Z) bounding-box padding in slices. "
                        "Kept small (default 1) because one Z slice = one video frame.")
    p.add_argument("--save_nifti",    action="store_true")
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--bbox_mode",   default="gt",
                   choices=["gt", "pet", "unet", "hybrid"],
                   help="Prompt source. 'gt'=oracle (dev only), others=auto.")
    p.add_argument("--pet_method",  default="base41",
                   choices=["base41", "nestle", "black", "daisne"],
                   help="PET thresholding strategy for 'pet'/'hybrid' modes.")
    p.add_argument("--proposal_model",  default=None,
                   help="Path to Small3DUNet .pt checkpoint "
                        "(required for 'unet'/'hybrid').")
    p.add_argument("--prob_threshold",  type=float, default=0.25,
                   help="UNet probability threshold (recall-biased default 0.25).")
    main(p.parse_args())
