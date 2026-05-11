"""
test_auto_prompting.py
======================
Visual smoke-test for the auto_prompting package.

Analogous to test_slicer.py: loads K random NPZ files from a val directory,
runs the enabled test modules, and saves one PNG figure per patient.

Tests
-----
A) PET proposals  (always)
   For each patient, one figure row per PET method (base41/nestle/black/daisne).
   Columns: CT at key_z, PET at key_z, PET mask overlay + bounding boxes.

B) Proposal network  (when --proposal_model is given)
   Extra row showing the UNet probability map and derived proposals.

C) AutoPrompter hybrid  (when --proposal_model is given)
   Extra row showing the hybrid proposals after PET∩UNet filtering.

Usage
-----
# PET-only test (no model needed)
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

# ── Repo root on path ────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE
for candidate in [_HERE, _HERE.parent]:
    if (candidate / "slicer.py").exists():
        _REPO_ROOT = candidate
        break
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from auto_prompting.pet_proposals import (
    get_pet_proposals, reconstruct_suv,
    base41_mask, nestle_mask, black_mask, daisne_mask,
)
from auto_prompting.auto_prompter import AutoPrompter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PET_METHODS = ["base41", "nestle", "black", "daisne"]

METHOD_COLORS = {
    "base41": "gold",
    "nestle": "deepskyblue",
    "black":  "tomato",
    "daisne": "limegreen",
}

GT_COLORS = {1: (1.0, 0.2, 0.2), 2: (0.2, 0.6, 1.0)}   # GTVp / GTVn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "ct_imgs":    data["ct_imgs"],
        "pet_imgs":   data["pet_imgs"],
        "gts":        data["gts"],
        "suv_max":    float(data["pet_suv_max"]) if "pet_suv_max" in data else None,
    }


def draw_boxes(ax, proposals: list, color: str, linewidth: float = 1.5) -> None:
    """Draw [x0, y0, x1, y1] bounding boxes on ax."""
    for p in proposals:
        x0, y0, x1, y1 = p["bbox_2d"]
        w, h = x1 - x0, y1 - y0
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=0.9,
        ))
        ax.text(x0, y0 - 2, f"#{p['component_id']} {p['voxel_count']:,}vox",
                color=color, fontsize=6, va="bottom")


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
    """Return the key_z of the largest proposal, or the volume midpoint."""
    return proposals[0]["z_mid"] if proposals else D // 2


# ---------------------------------------------------------------------------
# A) PET proposals figure
# ---------------------------------------------------------------------------

def make_pet_figure(patient_id: str, data: dict,
                    args: argparse.Namespace) -> plt.Figure:
    ct   = data["ct_imgs"]
    pet  = data["pet_imgs"]
    gts  = data["gts"]
    smx  = data["suv_max"]
    D    = ct.shape[0]

    n_rows = len(PET_METHODS)
    n_cols = 3   # CT | PET | mask+boxes
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5),
                             constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"PET proposals — {patient_id}  |  suv_max={smx:.2f if smx else 'N/A'}",
        color="white", fontsize=10,
    )

    for row, method in enumerate(PET_METHODS):
        props = get_pet_proposals(
            pet_uint8  = pet,
            suv_max    = smx,
            method     = method,
            slice_pad  = args.slice_pad,
            planar_pad = args.planar_pad,
        )
        z = best_z(props, D)
        color = METHOD_COLORS[method]

        # Threshold mask for visualisation (always re-derive on whatever data we have)
        if method == "base41" or smx is None:
            mask2d = base41_mask(pet)[z]
        else:
            suv = reconstruct_suv(pet, smx)
            mask2d = {
                "nestle": nestle_mask,
                "black":  black_mask,
                "daisne": daisne_mask,
            }[method](suv)[z]

        ax_ct, ax_pet, ax_mask = axes[row]
        for ax in (ax_ct, ax_pet, ax_mask):
            ax.set_facecolor("#1a1a1a")
            ax.axis("off")

        # CT
        ax_ct.imshow(ct[z], cmap="gray", interpolation="nearest")
        gt_overlay(ax_ct, gts[z])
        ax_ct.set_title(f"[{method}]  CT z={z}", color="white", fontsize=8)

        # PET
        ax_pet.imshow(pet[z], cmap="hot", interpolation="nearest")
        ax_pet.set_title("PET (uint8)", color="white", fontsize=8)

        # Mask + boxes
        ax_mask.imshow(ct[z], cmap="gray", interpolation="nearest")
        overlay = np.zeros((*mask2d.shape, 4), dtype=np.float32)
        overlay[mask2d, :3] = plt.cm.colors.to_rgb(color)
        overlay[mask2d,  3] = 0.45
        ax_mask.imshow(overlay, interpolation="nearest")
        draw_boxes(ax_mask, [p for p in props if p["z_mid"] == z], color)
        ax_mask.set_title(
            f"Mask + boxes  ({len(props)} proposals)", color="white", fontsize=8
        )

    return fig


# ---------------------------------------------------------------------------
# B) Proposal network figure
# ---------------------------------------------------------------------------

def make_net_figure(patient_id: str, data: dict,
                    model, device: str,
                    args: argparse.Namespace) -> plt.Figure:
    ct  = data["ct_imgs"]
    pet = data["pet_imgs"]
    gts = data["gts"]
    D   = ct.shape[0]

    x = torch.tensor(
        np.stack([ct.astype(np.float32) / 255.0,
                  pet.astype(np.float32) / 255.0], axis=0)
    ).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        prob = model(x)[0, 0].cpu().numpy()   # (D, H, W)

    thresholds = [0.15, 0.25, 0.35]
    n_rows, n_cols = len(thresholds), 3  # prob map | binary+boxes | CT+GT

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5),
                             constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"Proposal network — {patient_id}", color="white", fontsize=10)

    for row, thr in enumerate(thresholds):
        from auto_prompting.pet_proposals import mask_to_proposals
        props = mask_to_proposals(prob > thr,
                                  slice_pad=args.slice_pad,
                                  planar_pad=args.planar_pad)
        z = best_z(props, D)

        ax_prob, ax_bin, ax_ct = axes[row]
        for ax in (ax_prob, ax_bin, ax_ct):
            ax.set_facecolor("#1a1a1a")
            ax.axis("off")

        # Probability map
        ax_prob.imshow(prob[z], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax_prob.set_title(f"Prob map  thr={thr}  z={z}", color="white", fontsize=8)

        # Binary + boxes
        ax_bin.imshow(ct[z], cmap="gray", interpolation="nearest")
        bin_overlay = np.zeros((*prob[z].shape, 4), dtype=np.float32)
        bin_overlay[prob[z] > thr, :] = [1.0, 0.8, 0.0, 0.5]
        ax_bin.imshow(bin_overlay, interpolation="nearest")
        draw_boxes(ax_bin, [p for p in props if p["z_mid"] == z], "gold")
        ax_bin.set_title(f"{len(props)} proposals", color="white", fontsize=8)

        # CT + GT
        ax_ct.imshow(ct[z], cmap="gray", interpolation="nearest")
        gt_overlay(ax_ct, gts[z])
        ax_ct.set_title("CT + GT", color="white", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# C) AutoPrompter hybrid figure
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

    label_colors_box = {1: "tomato", 2: "deepskyblue"}
    label_names      = {1: "GTVp",   2: "GTVn"}

    for ax, (title, cmap, bg) in zip(axes, [
        ("CT  z={}".format(z), "gray", ct[z]),
        ("PET z={}".format(z), "hot",  pet[z]),
        ("Proposals + GT",     "gray", ct[z]),
    ]):
        ax.set_facecolor("#1a1a1a")
        ax.imshow(bg, cmap=cmap, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=8)
        ax.axis("off")

    # Overlay GT and boxes on third panel
    gt_overlay(axes[2], gts[z])
    for label_id, comps in components_per_label.items():
        c = label_colors_box[label_id]
        on_this_z = [p for p in comps if p["z_mid"] == z]
        draw_boxes(axes[2], on_this_z, c)
        if on_this_z:
            axes[2].set_title(
                f"Proposals + GT  [{label_names[1]}={len(components_per_label.get(1,[]))} "
                f"{label_names[2]}={len(components_per_label.get(2,[]))}]",
                color="white", fontsize=8,
            )

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

        # ── A: PET proposals ─────────────────────────────────────────────
        fig = make_pet_figure(pid, data, args)
        out = output_dir / f"{pid}_pet_proposals.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved → {out}")

        # ── B: Proposal network ──────────────────────────────────────────
        if model is not None:
            fig = make_net_figure(pid, data, model, device, args)
            out = output_dir / f"{pid}_proposal_net.png"
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
