#!/usr/bin/env python3
"""
ensemble_kfold_predictions.py
==============================
Majority-vote ensemble of per-fold predictions produced by test.py.

For each case, loads k prediction NIfTIs (one per fold model), computes a
per-voxel majority vote across the k segmentation maps, and writes the
ensembled prediction.

Usage
-----
  python npz_version/ensemble_kfold_predictions.py \\
      --fold_dirs \\
          /data/ethan/SwinCross/kfold_run/fold0/hecktor_inference \\
          /data/ethan/SwinCross/kfold_run/fold1/hecktor_inference \\
          /data/ethan/SwinCross/kfold_run/fold2/hecktor_inference \\
          /data/ethan/SwinCross/kfold_run/fold3/hecktor_inference \\
          /data/ethan/SwinCross/kfold_run/fold4/hecktor_inference \\
      --output_dir /data/ethan/SwinCross/kfold_run/ensemble \\
      --data_dir   /data/ethan/PP_hecktor2026_kfold_npz \\
      --json_list  dataset_swincross_2026kfold_full.json

Notes
-----
  • Majority vote: for each voxel the label that appears most often across the
    k folds is chosen.  Ties are broken by taking the label with the highest
    index (i.e. GTVn > GTVp > background) — conservative assumption is that
    a real structure exists when uncertain.

  • All fold predictions must share the same physical space (they do, since
    test.py writes in original CT space via inverse_transform_to_original_space).
    A shape-mismatch guard resamples outlier predictions to the majority shape.
"""

import argparse
import glob
import json
import os
import warnings
from collections import Counter

import numpy as np
import SimpleITK as sitk

warnings.filterwarnings("ignore")


def majority_vote(arrays: list) -> np.ndarray:
    """
    Per-voxel majority vote over a list of uint8 numpy arrays with the same shape.
    Ties broken in favour of the higher label index.
    """
    stack  = np.stack(arrays, axis=0)           # (k, R, A, S)
    result = np.zeros(arrays[0].shape, dtype=np.uint8)
    k      = stack.shape[0]

    # scipy.stats.mode is straightforward but slow on large volumes; this
    # numpy approach is equivalent and runs in one vectorised pass.
    unique_labels = np.unique(stack)
    vote_counts   = np.zeros((len(unique_labels),) + arrays[0].shape, dtype=np.int16)
    for i, lbl in enumerate(unique_labels):
        vote_counts[i] = (stack == lbl).sum(axis=0)

    # argmax over label axis; tie-break: np.argmax returns first max, so we
    # reverse the label order to prefer higher labels.
    vote_counts_rev   = vote_counts[::-1]
    unique_labels_rev = unique_labels[::-1]
    winner_idx        = np.argmax(vote_counts_rev, axis=0)
    result            = unique_labels_rev[winner_idx].astype(np.uint8)
    return result


def resample_to_reference(src: sitk.Image, ref: sitk.Image) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetOutputPixelType(sitk.sitkUInt8)
    return r.Execute(src)


def main() -> None:
    ap = argparse.ArgumentParser(description="Majority-vote ensemble of k-fold predictions")
    ap.add_argument("--fold_dirs",   nargs="+", required=True,
                    help="One directory per fold, each containing <case_id>_Pred.nii.gz files.")
    ap.add_argument("--output_dir",  required=True,
                    help="Where to write ensembled predictions.")
    ap.add_argument("--data_dir",    required=True,
                    help="NPZ data root (used to iterate the JSON case list).")
    ap.add_argument("--json_list",   required=True,
                    help="JSON file within data_dir that lists cases to ensemble.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k = len(args.fold_dirs)
    print(f"Ensembling {k} fold predictions → {args.output_dir}/")

    # Load case list
    with open(os.path.join(args.data_dir, args.json_list)) as f:
        dataset_json = json.load(f)
    all_entries = []
    for split in ("validation", "training", "testing"):
        all_entries.extend(dataset_json.get(split, []))

    found = missing = 0
    for idx, entry in enumerate(all_entries):
        case_id = entry.get("case_id") or os.path.basename(
            entry.get("npz", "unknown")).replace(".npz", "")

        output_path = os.path.join(args.output_dir, f"{case_id}_Pred.nii.gz")

        # Collect predictions from each fold
        fold_preds: list[sitk.Image] = []
        for fold_dir in args.fold_dirs:
            cands = glob.glob(os.path.join(fold_dir, f"{case_id}_Pred.nii.gz"))
            if cands:
                fold_preds.append(sitk.ReadImage(cands[0]))
            else:
                print(f"  ⚠  [{case_id}] fold prediction missing in {fold_dir}")

        if not fold_preds:
            print(f"  ❌ [{case_id}] No fold predictions found — skipping")
            missing += 1
            continue
        if len(fold_preds) < k:
            print(f"  ⚠  [{case_id}] Only {len(fold_preds)}/{k} folds available")

        # Align all predictions to the first fold's physical space
        reference = fold_preds[0]
        arrays    = [sitk.GetArrayFromImage(reference).astype(np.uint8)]
        for fp in fold_preds[1:]:
            if fp.GetSize() != reference.GetSize():
                print(f"     Resampling mismatched fold pred to reference space")
                fp = resample_to_reference(fp, reference)
            arrays.append(sitk.GetArrayFromImage(fp).astype(np.uint8))

        ensembled_arr = majority_vote(arrays)

        # Package as SimpleITK image with the reference's physical metadata
        ensembled_sitk = sitk.GetImageFromArray(ensembled_arr)
        ensembled_sitk.CopyInformation(reference)
        ensembled_sitk = sitk.Cast(ensembled_sitk, sitk.sitkUInt8)

        sitk.WriteImage(ensembled_sitk, output_path)
        found += 1
        labels_found = np.unique(ensembled_arr).tolist()
        print(f"  ✅ [{idx+1}/{len(all_entries)}] {case_id}  labels={labels_found}")

    print(f"\nEnsemble complete: {found} saved, {missing} skipped")


if __name__ == "__main__":
    main()
