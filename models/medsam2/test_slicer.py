"""
test_slicer.py
==============
Visual smoke-test for slicer.py.

For each of the k sampled patients:
  - Runs find_components() on the GT mask
  - Dynamically determines the optimal viewing axis (Z, Y, or X)
  - Renders a 5-slice sequence: Pad Min, Tight Min, Mid, Tight Max, Pad Max
  - Draws BOTH the Tight Bounding Box (dotted) and Padded Bounding Box (solid)
  - Saves one PNG summary figure per patient under --output_dir
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

from slicer import find_components

# ---------------------------------------------------------------------------
# Colour palette & Constants
# ---------------------------------------------------------------------------
LABEL_STYLE = {
    1: dict(name="GTVp", mask_color=(1.00, 0.20, 0.20), bbox_color="tomato"),
    2: dict(name="GTVn", mask_color=(0.20, 0.55, 1.00), bbox_color="deepskyblue"),
}

AXIS_NAMES = {0: "Z (Axial)", 1: "Y (Coronal)", 2: "X (Sagittal)"}
AXIS_LABELS = {0: "z", 1: "y", 2: "x"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "ct_imgs":  data["ct_imgs"],   
        "pet_imgs": data["pet_imgs"],  
        "gts":      data["gts"],       
        "spacing":  data["spacing"],   
    }

def make_rgba_overlay(mask_2d: np.ndarray, rgb: tuple, alpha: float = 0.45) -> np.ndarray:
    rgba = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
    rgba[mask_2d > 0, :3] = rgb
    rgba[mask_2d > 0,  3] = alpha
    return rgba

def draw_bbox(ax, bbox_2d: list, color: str, linestyle: str, linewidth: float, label: str = None) -> None:
    """Draw [col_min, row_min, col_max, row_max] as a rectangle on *ax*."""
    c0, r0, c1, r1 = bbox_2d
    w, h = c1 - c0, r1 - r0
    rect = mpatches.FancyBboxPatch(
        (c0, r0), w, h,
        boxstyle="square,pad=0",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        label=label,
        alpha=0.9
    )
    ax.add_patch(rect)

def extract_slice(vol_3d: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0: return vol_3d[idx, :, :]
    elif axis == 1: return vol_3d[:, idx, :]
    elif axis == 2: return vol_3d[:, :, idx]
    raise ValueError(f"Invalid axis: {axis}")

# ---------------------------------------------------------------------------
# Per-patient figure
# ---------------------------------------------------------------------------

def make_patient_figure(patient_id: str, data: dict, args: argparse.Namespace) -> plt.Figure:
    ct_imgs  = data["ct_imgs"]   
    pet_imgs = data["pet_imgs"]  
    gts      = data["gts"]       
    D, H, W  = ct_imgs.shape

    components_per_label = find_components(
        mask=gts,
        slice_pad=args.slice_pad,
        planar_pad=args.planar_pad,
        label_values=(1, 2),
    )

    num_mods   = 2 if args.modality == "both" else 1
    num_slices = 5 # Pad Min, Tight Min, Mid, Tight Max, Pad Max

    max_comps = max((len(comps) for comps in components_per_label.values()), default=1)
    n_cols = max_comps * num_mods * num_slices + 1
    n_rows = len(LABEL_STYLE)
    
    # Expand figure width significantly to handle 5 slices
    fig_w  = max(16, n_cols * 3.0)
    fig_h  = n_rows * 4.0

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#111111")

    title_text = (
        f"Slicer Debug — {patient_id}  |  Vol: {D}x{H}x{W}  |  Mod: {args.modality.upper()}  \n"
        f"Solid Line = Padded BBox (Model Input)   |   Dotted Line = Tight BBox (Actual Mask Bounds)"
    )
    fig.suptitle(title_text, color="white", fontsize=11, y=1.03)

    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.05)

    # ── overview column (right-most) ─────────────────────
    for row_idx, label_id in enumerate(sorted(LABEL_STYLE)):
        ax_ov = fig.add_subplot(grid[row_idx, -1])
        ax_ov.set_facecolor("#1a1a1a")
        ax_ov.set_title(f"{LABEL_STYLE[label_id]['name']} Overview", color="white", fontsize=8)
        ax_ov.set_xlabel("slice (z)", color="lightgray", fontsize=7)
        ax_ov.tick_params(colors="lightgray", labelsize=6, left=False, labelleft=False)
        for spine in ax_ov.spines.values(): spine.set_edgecolor("#444")

        label_area = np.array([(gts[z] == label_id).sum() for z in range(D)])
        ax_ov.fill_between(range(D), label_area, color=LABEL_STYLE[label_id]["mask_color"], alpha=0.25)
        ax_ov.plot(range(D), label_area, color=LABEL_STYLE[label_id]["mask_color"], lw=1)

        comps = components_per_label.get(label_id, [])
        for c_idx, comp in enumerate(comps):
            z0, z1 = comp["bbox_voxel"]["z"]
            ax_ov.axvspan(z0, z1, alpha=0.18, color=LABEL_STYLE[label_id]["bbox_color"])

    # ── per-component panels ──────────────────────────────────────────
    for row_idx, label_id in enumerate(sorted(LABEL_STYLE)):
        style = LABEL_STYLE[label_id]
        comps = components_per_label.get(label_id, [])

        if not comps:
            ax = fig.add_subplot(grid[row_idx, 0:n_cols - 1])
            ax.set_facecolor("#1a1a1a")
            ax.text(0.5, 0.5, f"{style['name']} — not present", ha="center", va="center", color="gray")
            ax.axis("off")
            continue

        for c_idx, comp in enumerate(comps):
            p_axis = comp["primary_axis"]
            mid_idx = comp["mid_slice"]
            bc = style["bbox_color"]
            
            # --- CALCULATE TIGHT BOUNDS ---
            mask_indices = np.where(comp["cropped_mask"])
            if len(mask_indices[0]) > 0:
                rel_z_min, rel_z_max = mask_indices[0].min(), mask_indices[0].max()
                rel_y_min, rel_y_max = mask_indices[1].min(), mask_indices[1].max()
                rel_x_min, rel_x_max = mask_indices[2].min(), mask_indices[2].max()

                base_z, base_y, base_x = comp["bbox_voxel"]["z"][0], comp["bbox_voxel"]["y"][0], comp["bbox_voxel"]["x"][0]

                tight_z_min, tight_z_max = base_z + rel_z_min, base_z + rel_z_max
                tight_y_min, tight_y_max = base_y + rel_y_min, base_y + rel_y_max
                tight_x_min, tight_x_max = base_x + rel_x_min, base_x + rel_x_max
            else:
                tight_z_min = tight_z_max = tight_y_min = tight_y_max = tight_x_min = tight_x_max = mid_idx

            # Get the exact slices along the viewing axis
            if p_axis == 0:
                tight_min, tight_max = tight_z_min, tight_z_max
                tight_bbox_2d = [tight_x_min, tight_y_min, tight_x_max + 1, tight_y_max + 1]
            elif p_axis == 1:
                tight_min, tight_max = tight_y_min, tight_y_max
                tight_bbox_2d = [tight_x_min, tight_z_min, tight_x_max + 1, tight_z_max + 1]
            else:
                tight_min, tight_max = tight_x_min, tight_x_max
                tight_bbox_2d = [tight_y_min, tight_z_min, tight_y_max + 1, tight_z_max + 1]

            pad_min = comp["bbox_voxel"][AXIS_LABELS[p_axis]][0]
            pad_max = comp["bbox_voxel"][AXIS_LABELS[p_axis]][1] - 1

            # 5 slices to show sequence
            slices_to_show = [
                (pad_min,   f"PAD Min {AXIS_LABELS[p_axis]}"),
                (tight_min, f"TIGHT Min {AXIS_LABELS[p_axis]}"),
                (mid_idx,   f"MID {AXIS_LABELS[p_axis]}"),
                (tight_max, f"TIGHT Max {AXIS_LABELS[p_axis]}"),
                (pad_max,   f"PAD Max {AXIS_LABELS[p_axis]}")
            ]

            comp_label = f"{style['name']} #{comp['component_id']} (vox:{comp['voxel_count']:,})"
            col_base = c_idx * (num_mods * num_slices)   
            max_vol_idx = ct_imgs.shape[p_axis] - 1

            for s_idx, (idx, slice_label) in enumerate(slices_to_show):
                idx = max(0, min(max_vol_idx, idx)) 

                gt_slice  = extract_slice(gts, p_axis, idx)
                ct_slice  = extract_slice(ct_imgs, p_axis, idx)
                pet_slice = extract_slice(pet_imgs, p_axis, idx)

                gt_label_slice = (gt_slice == label_id).astype(np.uint8)
                gt_rgba = make_rgba_overlay(gt_label_slice, style["mask_color"], alpha=0.55)

                mods_to_show = []
                if args.modality in ["ct", "both"]: mods_to_show.append((ct_slice, "CT"))
                if args.modality in ["pet", "both"]: mods_to_show.append((pet_slice, "PET"))

                for mod_idx, (img_slice, mod_name) in enumerate(mods_to_show):
                    col_idx = col_base + (s_idx * num_mods) + mod_idx
                    ax = fig.add_subplot(grid[row_idx, col_idx])
                    ax.set_facecolor("#1a1a1a")

                    ax.imshow(img_slice, cmap="gray", interpolation="nearest")
                    ax.imshow(gt_rgba,   interpolation="nearest")

                    # Draw Padded BBox (Solid)
                    draw_bbox(ax, comp["bbox_2d"], color=bc, linestyle="solid", linewidth=1.2)
                    
                    # Draw Tight BBox (Dotted)
                    draw_bbox(ax, tight_bbox_2d, color="white", linestyle=(0, (1, 1.5)), linewidth=1.2)

                    title = f"[{AXIS_NAMES[p_axis]}] {slice_label}: {idx} \n{mod_name}"
                    if s_idx == 2 and mod_idx == 0: 
                        title = f"{comp_label}\n{title}" # Only add comp info to the middle CT

                    ax.set_title(title, color="white", fontsize=7, pad=4)
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
    print(f"Testing slicer on {len(selected)} / {len(npz_files)} patients from {npz_dir}\n")

    for npz_path in selected:
        patient_id = npz_path.stem
        print(f"  [{patient_id}]")

        data = load_npz(npz_path)
        
        fig = make_patient_figure(patient_id, data, args)
        out_path = output_dir / f"{patient_id}_slicer_test.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved → {out_path}\n")

    print(f"Done. All figures written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual smoke-test for slicer.py extraction.")
    parser.add_argument("--npz_dir", type=str, default="/data/ethan/MedSAM2/hecktor_npz/val")
    parser.add_argument("--output_dir", type=str, default="/data/ethan/MedSAM2/slicer_test")
    parser.add_argument("--k", type=int, default=5, help="Number of patients to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--modality", type=str, choices=["ct", "pet", "both"], default="both")
    parser.add_argument("--slice_pad", type=int, default=1, help="Padding on the primary viewing axis.")
    parser.add_argument("--planar_pad", type=int, default=5, help="Padding on the planar dimensions.")
    main(parser.parse_args())
