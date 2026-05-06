"""
test_slicer.py
==============
Visual smoke-test for slicer.py.

For each of the k sampled patients in the val (or train) NPZ directory:
  - Runs find_components() on the GT mask
  - For every detected component, renders the key slice with
    - CT background, PET background, or both
    - GT mask overlay (semi-transparent)
    - Predicted bounding box
    - Component metadata (label, id, voxel count, z_mid)
  - Saves one PNG summary figure per patient under --output_dir

Usage
-----
python test_slicer.py \
    --npz_dir  /data/ethan/MedSAM2/hecktor_npz/val \
    --output_dir /data/ethan/MedSAM2/slicer_test \
    --k 5 \
    --seed 42
    --modality both
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── make sure repo root is on the path so slicer.py is importable ──────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slicer import find_components, scale_bbox_2d

# ---------------------------------------------------------------------------
# Colour palette – consistent across all figures
# ---------------------------------------------------------------------------
LABEL_STYLE = {
    1: dict(name="GTVp", mask_color=(1.00, 0.20, 0.20), bbox_color="tomato"),
    2: dict(name="GTVn", mask_color=(0.20, 0.55, 1.00), bbox_color="deepskyblue"),
}
# Distinct per-component box styles (dashed vs solid) so multiple components
# of the same label are still visually separable.
LINE_STYLES = ["solid", "dashed", "dotted", "dashdot"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "ct_imgs":  data["ct_imgs"],   # (D, H, W) uint8
        "pet_imgs": data["pet_imgs"],  # (D, H, W) uint8
        "gts":      data["gts"],       # (D, H, W) uint8
        "spacing":  data["spacing"],   # (3,) float64
    }


def make_rgba_overlay(mask_2d: np.ndarray, rgb: tuple, alpha: float = 0.45) -> np.ndarray:
    """Return an RGBA float array from a 2-D binary mask."""
    rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
    rgba[mask_2d > 0, :3] = rgb
    rgba[mask_2d > 0,  3] = alpha
    return rgba


def draw_bbox(ax, bbox_2d: np.ndarray, color: str, linestyle: str, label: str) -> None:
    """Draw [x_min, y_min, x_max, y_max] as a rectangle on *ax*."""
    x0, y0, x1, y1 = bbox_2d
    w, h = x1 - x0, y1 - y0
    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="square,pad=0",
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)
    # Corner label
    ax.text(
        x0 + 2, y0 + 2, label,
        fontsize=7, color=color,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.55, linewidth=0),
    )


# ---------------------------------------------------------------------------
# Per-patient figure
# ---------------------------------------------------------------------------

def make_patient_figure(patient_id: str, data: dict, modality: str = "both") -> plt.Figure:
    """
    Build a summary figure for one patient.

    Layout
    ------
    Two rows of panels, one per label (GTVp / GTVn).
    Each component occupies a pair of columns (CT | PET).
    An additional overview column on the right shows the full z-stack with
    component z-ranges marked.
    """
    ct_imgs  = data["ct_imgs"]   # (D, H, W)
    pet_imgs = data["pet_imgs"]  # (D, H, W)
    gts      = data["gts"]       # (D, H, W)
    D        = ct_imgs.shape[0]

    components_per_label = find_components(
        mask=gts,
        padding=5,
        label_values=(1, 2),
    )

    # Determine how many imaging columns per component
    num_mods = 2 if modality == "both" else 1

    # ── figure geometry ──────────────────────────────────────────────────
    # Find the maximum number of components across labels
    max_comps = max(
        (len(comps) for comps in components_per_label.values()),
        default=1,
    )
    # columns: max_comps × num_mods + 1 overview
    n_cols   = max_comps * num_mods + 1
    n_rows   = len(LABEL_STYLE)   # one row per label
    fig_w    = max(14, n_cols * 3.2)
    fig_h    = n_rows * 4.0

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#111111")

    title_text = (
        f"Slicer test — {patient_id}  "
        f"| D={D} slices  "
        f"| Modality: {modality.upper()}  "
        f"| labels: {sorted(components_per_label.keys())}"
    )
    fig.suptitle(title_text, color="white", fontsize=11, y=1.01)

    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.08)

    # ── overview column (right-most) for each row ─────────────────────
    for row_idx, label_id in enumerate(sorted(LABEL_STYLE)):
        ax_ov = fig.add_subplot(grid[row_idx, -1])
        ax_ov.set_facecolor("#1a1a1a")
        ax_ov.set_title(
            f"{LABEL_STYLE[label_id]['name']} — z overview",
            color="white", fontsize=8,
        )
        ax_ov.set_xlabel("slice (z)", color="lightgray", fontsize=7)
        ax_ov.set_ylabel("max cross-section (px²)", color="lightgray", fontsize=7)
        ax_ov.tick_params(colors="lightgray", labelsize=6)
        for spine in ax_ov.spines.values():
            spine.set_edgecolor("#444")

        # Background: area of label_id per slice
        label_area = np.array([(gts[z] == label_id).sum() for z in range(D)])
        ax_ov.fill_between(
            range(D), label_area,
            color=LABEL_STYLE[label_id]["mask_color"], alpha=0.25,
        )
        ax_ov.plot(range(D), label_area,
                   color=LABEL_STYLE[label_id]["mask_color"], lw=1)

        # Mark key slices for each component
        comps = components_per_label.get(label_id, [])
        for c_idx, comp in enumerate(comps):
            lskey   = LINE_STYLES[c_idx % len(LINE_STYLES)]
            bcolor  = LABEL_STYLE[label_id]["bbox_color"]
            z0, z1  = comp["bbox_voxel"]["z"]
            ax_ov.axvspan(z0, z1, alpha=0.18, color=bcolor)
            ax_ov.axvline(comp["z_mid"], color=bcolor, lw=1.5,
                          linestyle=lskey,
                          label=f"comp {comp['component_id']} z_mid={comp['z_mid']}")
        if comps:
            ax_ov.legend(fontsize=5.5, loc="upper right",
                         labelcolor="white", facecolor="#222", edgecolor="#555")

    # ── per-component panels ──────────────────────────────────────────
    for row_idx, label_id in enumerate(sorted(LABEL_STYLE)):
        style  = LABEL_STYLE[label_id]
        comps  = components_per_label.get(label_id, [])

        if not comps:
            # Empty label: grey placeholder
            ax = fig.add_subplot(grid[row_idx, 0:n_cols - 1])
            ax.set_facecolor("#1a1a1a")
            ax.text(0.5, 0.5,
                    f"{style['name']} — not present in this patient",
                    ha="center", va="center",
                    color="gray", fontsize=10, transform=ax.transAxes)
            ax.axis("off")
            continue

        for c_idx, comp in enumerate(comps):
            z    = comp["z_mid"]
            bbox = comp["bbox_2d"]    # [x_min, y_min, x_max, y_max]
            ls   = LINE_STYLES[c_idx % len(LINE_STYLES)]
            bc   = style["bbox_color"]

            gt_slice  = gts[z]
            ct_slice  = ct_imgs[z]
            pet_slice = pet_imgs[z]

            # Build 2-D GT mask for this label only
            gt_label_slice = (gt_slice == label_id).astype(np.uint8)
            gt_rgba = make_rgba_overlay(gt_label_slice, style["mask_color"], alpha=0.45)

            comp_label = (
                f"{style['name']} #{comp['component_id']}  "
                f"z={z}  vox={comp['voxel_count']:,}"
            )

            col_base = c_idx * num_mods   # offset based on how many modalities we render

            # Decide which images to loop over
            mods_to_show = []
            if modality in ["ct", "both"]:
                mods_to_show.append((ct_slice, "CT"))
            if modality in ["pet", "both"]:
                mods_to_show.append((pet_slice, "PET"))

            for mod_idx, (img_slice, mod_name) in enumerate(mods_to_show):
                ax = fig.add_subplot(grid[row_idx, col_base + mod_idx])
                ax.set_facecolor("#1a1a1a")

                ax.imshow(img_slice, cmap="gray", interpolation="nearest")
                ax.imshow(gt_rgba,   interpolation="nearest")

                draw_bbox(ax, bbox, color=bc, linestyle=ls, label=comp_label)

                title = f"{comp_label}\n[{mod_name}]"
                ax.set_title(title, color="white", fontsize=7, pad=3)
                ax.axis("off")

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
    print(f"Testing slicer on {len(selected)} / {len(npz_files)} patients "
          f"from {npz_dir}\n")

    for npz_path in selected:
        patient_id = npz_path.stem
        print(f"  [{patient_id}]")

        data = load_npz(npz_path)
        gts  = data["gts"]
        D    = gts.shape[0]

        # ── console summary ────────────────────────────────────────────
        components_per_label = find_components(
            mask=gts, padding=5, label_values=(1, 2)
        )
        if not components_per_label:
            print("    [WARN] no foreground labels found — skipping figure.")
            continue

        for label_id, comps in sorted(components_per_label.items()):
            name = LABEL_STYLE[label_id]["name"]
            print(f"    {name}: {len(comps)} component(s)")
            for comp in comps:
                z0, z1 = comp["bbox_voxel"]["z"]
                print(
                    f"      comp {comp['component_id']:2d} | "
                    f"voxels={comp['voxel_count']:6,} | "
                    f"z_range=[{z0},{z1}]  z_mid={comp['z_mid']} | "
                    f"bbox={comp['bbox_2d'].tolist()}"
                )

        # ── figure ────────────────────────────────────────────────────
        fig = make_patient_figure(patient_id, data, args.modality)
        out_path = output_dir / f"{patient_id}_slicer_test.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved → {out_path}\n")

    print(f"Done. All figures written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual smoke-test for slicer.py bounding-box extraction."
    )
    parser.add_argument(
        "--npz_dir", type=str,
        default="/data/ethan/MedSAM2/hecktor_npz/val",
        help="Directory containing prepared HECKTOR NPZ files.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data/ethan/MedSAM2/slicer_test",
        help="Where to save the output PNG figures.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of patients to sample and visualise.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for patient sampling.",
    )
    parser.add_argument(
        "--modality", type=str, choices=["ct", "pet", "both"], default="both",
        help="Which imaging modality to display the overlay on.",
    )
    main(parser.parse_args())
