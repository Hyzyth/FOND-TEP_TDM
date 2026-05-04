"""
evaluate_hecktor.py
===================
Compute Dice Similarity Coefficient (DSC) for GTVp (label 1) and GTVn (label 2)
from predicted NPZ files against ground-truth NPZ files.

GT label convention (matches prepare_hecktor_npz.py output):
    0 – background
    1 – GTVp  (primary tumour)
    2 – GTVn  (nodal tumour; may be absent → DSC reported as NaN and excluded)

Usage
-----
python inference/evaluate_hecktor.py \
    --pred_dir /data/ethan/MedSAM2/predictions/val \
    --gt_dir   /data/ethan/MedSAM2/hecktor_npz/val \
    --output   /data/ethan/MedSAM2/predictions/val/dsc_results.csv
"""

import argparse
import csv
import os
from glob import glob
from os.path import basename, join

import numpy as np
from tqdm import tqdm

LABEL_GTVp = 1
LABEL_GTVn = 2


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, gt: np.ndarray, label: int, smooth: float = 1e-6) -> float:
    """Compute DSC for a single binary label.

    Parameters
    ----------
    pred, gt : np.ndarray
        Integer label arrays of identical shape.
    label : int
        Class ID to evaluate.
    smooth : float
        Laplace smoothing term (avoids division by zero when both are empty).

    Returns
    -------
    float  DSC in [0, 1].  Returns NaN when the GT is entirely absent.
    """
    p = (pred == label).astype(np.float32)
    g = (gt   == label).astype(np.float32)

    if g.sum() == 0:
        return float("nan")

    intersection = (p * g).sum()
    return float((2.0 * intersection + smooth) / (p.sum() + g.sum() + smooth))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    pred_files = sorted(glob(join(args.pred_dir, "*.npz")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction NPZ files found in {args.pred_dir}")

    rows = []
    dsc_gtvp_all: list[float] = []
    dsc_gtvn_all: list[float] = []

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        name    = basename(pred_path)
        gt_path = join(args.gt_dir, name)

        if not os.path.exists(gt_path):
            print(f"  [SKIP] GT not found for {name}")
            continue

        pred_data = np.load(pred_path, allow_pickle=True)
        gt_data   = np.load(gt_path,   allow_pickle=True)

        # Prediction NPZ contains key "segs"
        segs = pred_data["segs"]

        # GT NPZ contains key "gts" (produced by prepare_hecktor_npz.py)
        if "gts" in gt_data:
            gts = gt_data["gts"]
        else:
            # Fallback: some older exports may use "segs"
            gts = gt_data["segs"]

        if segs.shape != gts.shape:
            print(
                f"  [WARN] Shape mismatch for {name}: "
                f"pred={segs.shape}, gt={gts.shape} – skipping."
            )
            continue

        dsc_p = dice(segs, gts, LABEL_GTVp)
        dsc_n = dice(segs, gts, LABEL_GTVn)

        if not np.isnan(dsc_p):
            dsc_gtvp_all.append(dsc_p)
        if not np.isnan(dsc_n):
            dsc_gtvn_all.append(dsc_n)

        rows.append({
            "patient":  name.replace(".npz", ""),
            "dsc_gtvp": f"{dsc_p:.4f}" if not np.isnan(dsc_p) else "N/A",
            "dsc_gtvn": f"{dsc_n:.4f}" if not np.isnan(dsc_n) else "N/A",
        })

    # ── Aggregate ────────────────────────────────────────────────────────
    mean_p       = float(np.mean(dsc_gtvp_all)) if dsc_gtvp_all else float("nan")
    mean_n       = float(np.mean(dsc_gtvn_all)) if dsc_gtvn_all else float("nan")
    valid_means  = [x for x in [mean_p, mean_n] if not np.isnan(x)]
    mean_overall = float(np.mean(valid_means)) if valid_means else float("nan")

    rows.append({
        "patient":  "MEAN",
        "dsc_gtvp": f"{mean_p:.4f}",
        "dsc_gtvn": f"{mean_n:.4f}",
    })

    # ── Write CSV ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "dsc_gtvp", "dsc_gtvn"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*44}")
    print(f"  DSC GTVp  : {mean_p:.4f}  (n={len(dsc_gtvp_all)})")
    print(f"  DSC GTVn  : {mean_n:.4f}  (n={len(dsc_gtvn_all)})")
    print(f"  DSC mean  : {mean_overall:.4f}")
    print(f"{'='*44}")
    print(f"Results saved → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MedSAM2 HECKTOR predictions with DSC."
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing predicted NPZ files (key 'segs').",
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory containing ground-truth NPZ files (key 'gts').",
    )
    parser.add_argument(
        "--output", type=str, default="dsc_results.csv",
        help="Path for the output CSV file.",
    )
    main(parser.parse_args())
