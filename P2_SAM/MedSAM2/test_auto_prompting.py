"""
test_auto_prompting.py
======================
Visual smoke-test for the auto_prompting package.

Tests
-----
A) PET parameter sweeps  (always, one figure per method per patient)
   Each figure has one row per parameter variant, columns: CT | PET | mask+boxes.
   Variants sweep the core parameters of each thresholding method:
     base41 : pct   from 0.20 → 0.60
     nestle : alpha from 0.05 → 0.30
     black  : (alpha, beta) combinations calibrated on literature
     daisne : (a, b) combinations around the published values

B) Proposal network thresholds  (when --proposal_model is given)
   Dense grid from 0.02 to 0.95: quickly reveals where the model's
   operating point should be.

C) AutoPrompter hybrid  (when --proposal_model is given)
   Single figure per patient showing the default hybrid result.

Usage
-----
# PET sweeps only (no model needed)
python test_auto_prompting.py \\
    --npz_dir /data/ethan/MedSAM2/hecktor_npz/val \\
    --output_dir /data/ethan/MedSAM2/auto_prompt_test \\
    --k 5

# Full test with proposal network
python test_auto_prompting.py \\
    --npz_dir /data/ethan/MedSAM2/hecktor_npz/val \\
    --output_dir /data/ethan/MedSAM2/auto_prompt_test \\
    --k 5 \\
    --proposal_model /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── Repo root on path ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for candidate in [_HERE, _HERE.parent]:
    if (candidate / "slicer.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from auto_prompting.pet_proposal import (
    get_pet_proposals, reconstruct_suv,
    base41_mask, nestle_mask, black_mask, daisne_mask,
)
from auto_prompting.auto_prompter import AutoPrompter

# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------

# Each entry: (label_for_figure, method_kwargs_dict)
PET_SWEEPS = {
    "base41": [
        (f"pct={p:.2f}", {"pct": p})
        for p in [0.20, 0.25, 0.30, 0.35, 0.41, 0.45, 0.50, 0.55, 0.60]
    ],
    "nestle": [
        (f"alpha={a:.2f}", {"alpha": a})
        for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ],
    "black": [
        (f"α={ab[0]:.3f} β={ab[1]:.3f}", {"alpha": ab[0], "beta": ab[1]})
        for ab in [
            (0.200, 0.400),
            (0.250, 0.500),
            (0.307, 0.588),   # published default
            (0.307, 0.700),
            (0.350, 0.600),
            (0.350, 0.700),
        ]
    ],
    "daisne": [
        (f"a={ab[0]:.1f} b={ab[1]:.1f}", {"a": ab[0], "b": ab[1]})
        for ab in [
            (25.0, 60.0),
            (28.0, 70.0),
            (31.3, 77.7),   # published default
            (35.0, 85.0),
            (40.0, 90.0),
            (45.0, 100.0),
        ]
    ],
}

# Dense threshold grid for the proposal network
UNET_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                   0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Color per method (for bounding boxes)
METHOD_COLORS = {
    "base41": "gold",
    "nestle": "deepskyblue",
    "black":  "tomato",
    "daisne": "limegreen",
}
GT_COLORS = {1: (1.0, 0.2, 0.2), 2: (0.2, 0.6, 1.0)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "ct_imgs":  data["ct_imgs"],
        "pet_imgs": data["pet_imgs"],
        "gts":      data["gts"],
        "suv_max":  float(data["pet_suv_max"]) if "pet_suv_max" in data else None,
    }


def draw_boxes(ax, proposals: list, color: str, lw: float = 1.5) -> None:
    for p in proposals:
        x0, y0, x1, y1 = p["bbox_2d"]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=0", linewidth=lw,
            edgecolor=color, facecolor="none", alpha=0.9,
        ))
        ax.text(x0, y0 - 2, f"#{p['component_id']} {p['voxel_count']:,}v",
                color=color, fontsize=5, va="bottom")


def gt_overlay(ax, gts_slice: np.ndarray, alpha: float = 0.45) -> None:
    for label_id, rgb in GT_COLORS.items():
        m = (gts_slice == label_id).astype(np.float32)
        if m.sum() == 0:
            continue
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = m * alpha
        ax.imshow(rgba, interpolation="nearest")


def best_z(proposals: list, D: int) -> int:
    return proposals[0]["z_mid"] if proposals else D // 2


def _dark_ax(ax):
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")


# ---------------------------------------------------------------------------
# A) PET parameter sweep — one figure per method
# ---------------------------------------------------------------------------

def make_pet_sweep_figure(patient_id: str, data: dict,
                           method: str, args: argparse.Namespace) -> plt.Figure:
    ct   = data["ct_imgs"]
    pet  = data["pet_imgs"]
    gts  = data["gts"]
    smx  = data["suv_max"]
    D    = ct.shape[0]

    variants = PET_SWEEPS[method]
    n_rows   = len(variants)
    color    = METHOD_COLORS[method]

    # Pre-compute SUV once for SUV-based methods
    suv = None
    if method != "base41" and smx is not None:
        suv = reconstruct_suv(pet, smx)

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(12, n_rows * 3.0),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("#111111")
    suv_txt = f"{smx:.2f}" if smx else "N/A"
    fig.suptitle(f"{method} sweep — {patient_id}  |  suv_max={suv_txt}",
                 color="white", fontsize=10)

    for row, (label, kwargs) in enumerate(variants):
        # Compute thresholded mask for visualisation
        if method == "base41":
            mask_vol = base41_mask(pet, **kwargs)
        elif suv is None:
            mask_vol = base41_mask(pet)   # fallback if suv unavailable
        elif method == "nestle":
            mask_vol = nestle_mask(suv, **kwargs)
        elif method == "black":
            mask_vol = black_mask(suv, **kwargs)
        else:  # daisne
            mask_vol = daisne_mask(suv, **kwargs)

        props = get_pet_proposals(
            pet_uint8  = pet,
            suv_max    = smx,
            method     = method,
            slice_pad  = args.slice_pad,
            planar_pad = args.planar_pad,
            **kwargs,
        )
        z = best_z(props, D)

        ax_ct, ax_pet, ax_mask = axes[row]
        for ax in (ax_ct, ax_pet, ax_mask):
            _dark_ax(ax)

        ax_ct.imshow(ct[z], cmap="gray", interpolation="nearest")
        gt_overlay(ax_ct, gts[z])
        ax_ct.set_title(f"{label}  |  z={z}", color="white", fontsize=7)

        ax_pet.imshow(pet[z], cmap="hot", interpolation="nearest")
        ax_pet.set_title("PET", color="white", fontsize=7)

        # Mask overlay + boxes
        ax_mask.imshow(ct[z], cmap="gray", interpolation="nearest")
        m2d = mask_vol[z]
        ov = np.zeros((*m2d.shape, 4), dtype=np.float32)
        try:
            ov[m2d, :3] = plt.cm.colors.to_rgb(color)
        except Exception:
            ov[m2d, :3] = [1.0, 0.8, 0.0]
        ov[m2d, 3] = 0.45
        ax_mask.imshow(ov, interpolation="nearest")
        draw_boxes(ax_mask, [p for p in props if p["z_mid"] == z], color)
        ax_mask.set_title(f"{len(props)} proposals", color="white", fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# B) Proposal network — dense threshold sweep
# ---------------------------------------------------------------------------

def make_net_sweep_figure(patient_id: str, data: dict,
                           model, device: str,
                           args: argparse.Namespace) -> plt.Figure:
    ct  = data["ct_imgs"]
    pet = data["pet_imgs"]
    gts = data["gts"]

    x = torch.tensor(
        np.stack([ct.astype(np.float32) / 255.0,
                  pet.astype(np.float32) / 255.0], axis=0)
    ).unsqueeze(0).float().to(device)

    # --- FIX: Pad to multiple of 16 ---
    _, _, D, H, W = x.shape
    pad_d = (16 - (D % 16)) % 16
    pad_h = (16 - (H % 16)) % 16
    pad_w = (16 - (W % 16)) % 16
    
    # F.pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    model.eval()
    with torch.no_grad():
        # Run model and crop back to original (D, H, W)
        prob = model(x)[0, 0, :D, :H, :W].cpu().numpy()

    thresholds = UNET_THRESHOLDS
    n_rows = len(thresholds)

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(12, n_rows * 2.6),
                             constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"Proposal network threshold sweep — {patient_id}",
                 color="white", fontsize=10)

    from auto_prompting.pet_proposal import mask_to_proposals

    for row, thr in enumerate(thresholds):
        props = mask_to_proposals(prob > thr,
                                  slice_pad=args.slice_pad,
                                  planar_pad=args.planar_pad)
        z = best_z(props, D)

        ax_prob, ax_bin, ax_ct = axes[row]
        for ax in (ax_prob, ax_bin, ax_ct):
            _dark_ax(ax)

        ax_prob.imshow(prob[z], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax_prob.set_title(f"thr={thr:.2f}  z={z}", color="white", fontsize=7)

        ax_bin.imshow(ct[z], cmap="gray", interpolation="nearest")
        ov = np.zeros((*prob[z].shape, 4), dtype=np.float32)
        ov[prob[z] > thr] = [1.0, 0.8, 0.0, 0.5]
        ax_bin.imshow(ov, interpolation="nearest")
        draw_boxes(ax_bin, [p for p in props if p["z_mid"] == z], "gold")
        ax_bin.set_title(f"{len(props)} proposals", color="white", fontsize=7)

        ax_ct.imshow(ct[z], cmap="gray", interpolation="nearest")
        gt_overlay(ax_ct, gts[z])
        ax_ct.set_title("CT + GT", color="white", fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# C) Hybrid overview (single default run)
# ---------------------------------------------------------------------------

def make_hybrid_figure(patient_id: str, data: dict,
                        prompter: AutoPrompter,
                        args: argparse.Namespace) -> plt.Figure:
    ct  = data["ct_imgs"]
    pet = data["pet_imgs"]
    gts = data["gts"]
    D   = ct.shape[0]

    components_per_label = prompter.get_proposals(ct, pet, data["suv_max"])
    all_props = [p for comps in components_per_label.values() for p in comps]
    z = best_z(all_props, D)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Hybrid ({prompter.pet_method}) — {patient_id}  "
        f"| GTVp={len(components_per_label.get(1,[]))}  "
        f"GTVn={len(components_per_label.get(2,[]))}",
        color="white", fontsize=10,
    )
    for ax, (title, cmap, bg) in zip(axes, [
        (f"CT z={z}",  "gray", ct[z]),
        (f"PET z={z}", "hot",  pet[z]),
        ("Proposals + GT", "gray", ct[z]),
    ]):
        _dark_ax(ax)
        ax.imshow(bg, cmap=cmap, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=8)

    gt_overlay(axes[2], gts[z])
    label_colors = {1: "tomato", 2: "deepskyblue"}
    for label_id, comps in components_per_label.items():
        draw_boxes(axes[2], [p for p in comps if p["z_mid"] == z],
                   label_colors[label_id])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    npz_dir    = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")

    random.seed(args.seed)
    selected = random.sample(npz_files, min(args.k, len(npz_files)))
    print(f"Testing auto_prompting on {len(selected)}/{len(npz_files)} "
          f"patients from {npz_dir}\n")

    # Load model once if provided
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.proposal_model:
        from auto_prompting.proposal_net import Small3DUNet
        print(f"Loading proposal network: {args.proposal_model}")
        model = Small3DUNet.load(args.proposal_model, device=device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}  ({n_params/1e6:.2f}M)\n")

    # Build hybrid prompter (one per PET method so we can vary it)
    hybrid_prompter = None
    if model is not None:
        hybrid_prompter = AutoPrompter(
            method         = "hybrid",
            model_path     = args.proposal_model,
            pet_method     = "base41",   # shown in figure title
            device         = device,
            prob_threshold = args.prob_threshold,
            slice_pad      = args.slice_pad,
            planar_pad     = args.planar_pad,
        )

    for npz_path in selected:
        pid  = npz_path.stem
        print(f"  [{pid}]")
        data = load_npz(npz_path)

        # ── A: PET parameter sweeps ──────────────────────────────────────
        for method in PET_SWEEPS:
            if method != "base41" and data["suv_max"] is None:
                print(f"    [SKIP] {method} — no suv_max in NPZ")
                continue
            fig = make_pet_sweep_figure(pid, data, method, args)
            out = output_dir / f"{pid}_pet_{method}_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved → {out}")

        # ── B: Proposal network sweep ────────────────────────────────────
        if model is not None:
            fig = make_net_sweep_figure(pid, data, model, device, args)
            out = output_dir / f"{pid}_proposal_net_sweep.png"
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved → {out}")

        # ── C: Hybrid ────────────────────────────────────────────────────
        if hybrid_prompter is not None:
            fig = make_hybrid_figure(pid, data, hybrid_prompter, args)
            out = output_dir / f"{pid}_hybrid.png"
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved → {out}")

        print()

    print(f"Done. All figures written to {output_dir}")
    print()
    print("Output naming:")
    print("  {pid}_pet_{method}_sweep.png   — Test A: PET parameter sweep per method")
    if model is not None:
        print("  {pid}_proposal_net_sweep.png   — Test B: UNet density threshold sweep")
        print("  {pid}_hybrid.png               — Test C: Hybrid default result")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visual smoke-test for the auto_prompting package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz_dir",    default="/data/ethan/MedSAM2/hecktor_npz/val")
    p.add_argument("--output_dir", default="/data/ethan/MedSAM2/auto_prompt_test")
    p.add_argument("--k",          type=int,   default=5, help="Patients to sample.")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--proposal_model", default=None,
                   help="Path to Small3DUNet .pt checkpoint. "
                        "Enables tests B (net) and C (hybrid). "
                        "Omit to run PET-only test A.")
    p.add_argument("--prob_threshold", type=float, default=0.25)
    p.add_argument("--slice_pad",  type=int,   default=1)
    p.add_argument("--planar_pad", type=int,   default=5)
    main(p.parse_args())
