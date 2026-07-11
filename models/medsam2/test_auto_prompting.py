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
    get_pet_proposals, reconstruct_suv, mask_to_proposals,
    base41_mask, nestle_mask, black_mask, daisne_mask,
)
from auto_prompting.auto_prompter import AutoPrompter

# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------

# Each entry: (label_for_figure, method_kwargs_dict)
PET_SWEEPS = {
    "base41": [(f"pct={p:.2f}", {"pct": p}) for p in [0.20, 0.30, 0.41, 0.50, 0.60]],
    "nestle": [(f"alpha={a:.2f}", {"alpha": a}) for a in [0.05, 0.15, 0.25, 0.30]],
    "black":  [(f"α={ab[0]:.3f} β={ab[1]:.3f}", {"alpha": ab[0], "beta": ab[1]}) for ab in [(0.250, 0.500), (0.307, 0.588), (0.350, 0.700)]],
    "daisne": [(f"a={ab[0]:.1f} b={ab[1]:.1f}", {"a": ab[0], "b": ab[1]}) for ab in [(25.0, 60.0), (31.3, 77.7), (40.0, 90.0)]],
}
UNET_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

METHOD_COLORS = {"base41": "gold", "nestle": "deepskyblue", "black": "tomato", "daisne": "limegreen"}
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


def gt_overlay(ax, gts_slice: np.ndarray, alpha: float = 0.35) -> None:
    for label_id, rgb in GT_COLORS.items():
        m = (gts_slice == label_id).astype(np.float32)
        if m.sum() == 0: continue
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = m * alpha
        ax.imshow(rgba, interpolation="nearest")


def _dark_ax(ax):
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")


def plot_tumor_sequence(fig, grid, row, col_base, ct, gts, mask_vol, p, color):
    """Plots Min Z, Mid Z, and Max Z for a given tumor proposal dict."""
    z0, z1, y0, y1, x0, x1 = p["bbox_3d"]
    z_mid = p["z_mid"]
    z_max = max(z0, z1 - 1)

    slices = [(z0, f"Min Z: {z0}"), (z_mid, f"Mid Z: {z_mid}"), (z_max, f"Max Z: {z_max}")]
    edge_color = color if isinstance(color, str) else plt.cm.colors.to_hex(color)

    for s_idx, (z, title) in enumerate(slices):
        ax = fig.add_subplot(grid[row, col_base + s_idx])
        _dark_ax(ax)
        ax.imshow(ct[z], cmap="gray", interpolation="nearest")
        gt_overlay(ax, gts[z])

        if mask_vol is not None:
            m2d = mask_vol[z]
            ov = np.zeros((*m2d.shape, 4), dtype=np.float32)
            try:
                rgb = plt.cm.colors.to_rgb(color)
            except Exception:
                rgb = [1.0, 0.8, 0.0]
            ov[m2d > 0, :3] = rgb
            ov[m2d > 0, 3] = 0.45
            ax.imshow(ov, interpolation="nearest")

        w, h = x1 - x0, y1 - y0
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h, boxstyle="square,pad=0",
            linewidth=1.5, edgecolor=edge_color, facecolor="none", alpha=0.9
        ))

        vc = p.get("voxel_count", 0)
        cid = p.get("component_id", "?")
        full_title = f"#{cid} ({vc:,}v)\n{title}" if s_idx == 1 else title
        ax.set_title(full_title, color="white", fontsize=8)


# ---------------------------------------------------------------------------
# A) PET parameter sweep
# ---------------------------------------------------------------------------

def make_pet_sweep_figure(patient_id: str, data: dict, method: str, args: argparse.Namespace) -> plt.Figure:
    ct, pet, gts, smx = data["ct_imgs"], data["pet_imgs"], data["gts"], data["suv_max"]
    variants = PET_SWEEPS[method]
    color = METHOD_COLORS[method]

    suv = reconstruct_suv(pet, smx) if (method != "base41" and smx is not None) else None

    # Pre-compute to determine max tumors for grid sizing
    sweep_data = []
    max_tumors = 1
    for label, kwargs in variants:
        if method == "base41": mask_vol = base41_mask(pet, **kwargs)
        elif method == "nestle": mask_vol = nestle_mask(suv, **kwargs)
        elif method == "black": mask_vol = black_mask(suv, **kwargs)
        else: mask_vol = daisne_mask(suv, **kwargs)

        props = get_pet_proposals(pet, smx, method, slice_pad=args.slice_pad, planar_pad=args.planar_pad, **kwargs)
        max_tumors = max(max_tumors, len(props))
        sweep_data.append((label, mask_vol, props))

    max_tumors = min(max_tumors, 4) # Cap width
    slices_per_tumor = 3
    n_rows = len(variants)
    n_cols = 1 + (max_tumors * slices_per_tumor)

    fig = plt.figure(figsize=(max(8, n_cols * 2.5), n_rows * 3.0), constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"{method} sweep — {patient_id} | suv_max={smx:.2f}" if smx else f"{method} sweep — {patient_id}", color="white", fontsize=11)
    grid = fig.add_gridspec(n_rows, n_cols, wspace=0.05, hspace=0.3)

    for row, (label, mask_vol, props) in enumerate(sweep_data):
        ax_info = fig.add_subplot(grid[row, 0])
        _dark_ax(ax_info)
        ax_info.text(0.5, 0.5, f"{label}\n\n{len(props)} tumors", color=color, ha="center", va="center", fontsize=9, fontweight="bold")

        for t_idx, p in enumerate(props[:max_tumors]):
            plot_tumor_sequence(fig, grid, row, 1 + t_idx * slices_per_tumor, ct, gts, mask_vol, p, color)

    return fig


# ---------------------------------------------------------------------------
# B) Proposal network — dense threshold sweep
# ---------------------------------------------------------------------------

def make_net_sweep_figure(patient_id: str, data: dict, model, device: str, args: argparse.Namespace) -> plt.Figure:
    ct, pet, gts = data["ct_imgs"], data["pet_imgs"], data["gts"]

    x = torch.tensor(np.stack([ct.astype(np.float32)/255., pet.astype(np.float32)/255.], axis=0)).unsqueeze(0).float().to(device)
    _, _, D, H, W = x.shape
    pad_d, pad_h, pad_w = (16 - (D%16))%16, (16 - (H%16))%16, (16 - (W%16))%16
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    model.eval()
    with torch.no_grad():
        prob = model(x)[0, 0, :D, :H, :W].cpu().numpy()

    sweep_data, max_tumors = [], 1
    for thr in UNET_THRESHOLDS:
        props = mask_to_proposals(prob > thr, slice_pad=args.slice_pad, planar_pad=args.planar_pad)
        max_tumors = max(max_tumors, len(props))
        sweep_data.append((thr, props))

    max_tumors = min(max_tumors, 4)
    n_rows, n_cols = len(UNET_THRESHOLDS), 1 + (max_tumors * 3)
    
    fig = plt.figure(figsize=(max(8, n_cols * 2.5), n_rows * 3.0), constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"UNet sweep — {patient_id}", color="white", fontsize=11)
    grid = fig.add_gridspec(n_rows, n_cols, wspace=0.05, hspace=0.3)

    for row, (thr, props) in enumerate(sweep_data):
        ax_info = fig.add_subplot(grid[row, 0])
        _dark_ax(ax_info)
        ax_info.text(0.5, 0.5, f"thr={thr:.2f}\n\n{len(props)} tumors", color="gold", ha="center", va="center", fontsize=9, fontweight="bold")

        for t_idx, p in enumerate(props[:max_tumors]):
            plot_tumor_sequence(fig, grid, row, 1 + t_idx * 3, ct, gts, prob > thr, p, "gold")

    return fig


# ---------------------------------------------------------------------------
# C) Hybrid overview
# ---------------------------------------------------------------------------

def make_hybrid_figure(patient_id: str, data: dict, prompter: AutoPrompter, args: argparse.Namespace) -> plt.Figure:
    ct, gts = data["ct_imgs"], data["gts"]
    components_per_label = prompter.get_proposals(ct, data["pet_imgs"], data["suv_max"])

    max_tumors = max((len(comps) for comps in components_per_label.values()), default=1)
    max_tumors = min(max_tumors, 5)
    n_rows, n_cols = 2, 1 + (max_tumors * 3)

    fig = plt.figure(figsize=(max(8, n_cols * 2.5), n_rows * 3.5), constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"Hybrid Auto-Prompts ({prompter.pet_method}) — {patient_id}", color="white", fontsize=11)
    grid = fig.add_gridspec(n_rows, n_cols, wspace=0.05, hspace=0.3)

    labels = {1: "GTVp (Primary)", 2: "GTVn (Nodal)"}
    for row, label_id in enumerate([1, 2]):
        props = components_per_label.get(label_id, [])
        color = GT_COLORS[label_id]
        
        ax_info = fig.add_subplot(grid[row, 0])
        _dark_ax(ax_info)
        ax_info.text(0.5, 0.5, f"{labels[label_id]}\n\n{len(props)} tumors", color=color, ha="center", va="center", fontsize=10, fontweight="bold")

        for t_idx, p in enumerate(props[:max_tumors]):
            plot_tumor_sequence(fig, grid, row, 1 + t_idx * 3, ct, gts, None, p, color)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("*.npz"))
    random.seed(args.seed)
    selected = random.sample(npz_files, min(args.k, len(npz_files)))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if args.proposal_model:
        from auto_prompting.proposal_net import Small3DUNet
        model = Small3DUNet.load(args.proposal_model, device=device)
        
    hybrid_prompter = AutoPrompter("hybrid", args.proposal_model, "base41", device, args.prob_threshold, 
                                   slice_pad=args.slice_pad, planar_pad=args.planar_pad) if model else None

    for npz_path in selected:
        pid = npz_path.stem
        print(f"  [{pid}]")
        data = load_npz(npz_path)

        for method in PET_SWEEPS:
            if method != "base41" and data["suv_max"] is None: 
                print(f"    [SKIP] {method} — no suv_max in NPZ")
                continue
            fig = make_pet_sweep_figure(pid, data, method, args)
            out = output_dir / f"{pid}_pet_{method}_sweep.png"
            fig.savefig(output_dir / f"{pid}_pet_{method}_sweep.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved → {out}")

        if model:
            fig = make_net_sweep_figure(pid, data, model, device, args)
            out = output_dir / f"{pid}_proposal_net_sweep.png"
            fig.savefig(output_dir / f"{pid}_proposal_net_sweep.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved → {out}")

        if hybrid_prompter:
            fig = make_hybrid_figure(pid, data, hybrid_prompter, args)
            out = output_dir / f"{pid}_hybrid.png"
            fig.savefig(output_dir / f"{pid}_hybrid.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--npz_dir", default="/data/ethan/MedSAM2/hecktor_npz/val")
    p.add_argument("--output_dir", default="/data/ethan/MedSAM2/auto_prompt_test")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proposal_model", default=None)
    p.add_argument("--prob_threshold", type=float, default=0.25)
    p.add_argument("--slice_pad", type=int, default=1)
    p.add_argument("--planar_pad", type=int, default=5)
    main(p.parse_args())
