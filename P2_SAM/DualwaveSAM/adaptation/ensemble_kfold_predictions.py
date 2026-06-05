"""
ensemble_kfold_predictions.py  —  DualwaveSAM k-fold majority-vote ensemble
============================================================================

Usage:
  python {folder}/ensemble_kfold_predictions.py \\
      --fold_dirs \\
          /data/ethan/DualwaveSAM3c/kfold_run/fold0/hecktor_TEST_vault \\
          /data/ethan/DualwaveSAM3c/kfold_run/fold1/hecktor_TEST_vault \\
          ... \\
      --output_dir /data/ethan/DualwaveSAM3c/kfold_ensemble/hecktor_TEST_vault \\
      --data_dir   /data/ethan/PP_hecktor2026_kfold_npz \\
      --json_list  dataset_swincross_2026kfold_test.json
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import SimpleITK as sitk

warnings.filterwarnings("ignore")


def majority_vote(arrays: list) -> np.ndarray:
    """Per-voxel majority vote; ties broken in favour of higher label index."""
    stack         = np.stack(arrays, axis=0)        # (k, ...)
    unique_labels = np.unique(stack)
    vote_counts   = np.zeros((len(unique_labels),) + arrays[0].shape, dtype=np.int16)
    for i, lbl in enumerate(unique_labels):
        vote_counts[i] = (stack == lbl).sum(axis=0)
    # Reverse so argmax prefers higher labels on ties
    vote_counts_rev   = vote_counts[::-1]
    unique_labels_rev = unique_labels[::-1]
    winner_idx        = np.argmax(vote_counts_rev, axis=0)
    return unique_labels_rev[winner_idx].astype(np.uint8)


def resample_to_reference(src: sitk.Image, ref: sitk.Image) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetOutputPixelType(sitk.sitkUInt8)
    return r.Execute(src)

def check_geometry_match(img1: sitk.Image, img2: sitk.Image, tol=1e-5) -> bool:
    if img1.GetSize() != img2.GetSize(): return False
    
    # Check spacing and direction with a small float tolerance
    space_diff = np.max(np.abs(np.array(img1.GetSpacing()) - np.array(img2.GetSpacing())))
    dir_diff = np.max(np.abs(np.array(img1.GetDirection()) - np.array(img2.GetDirection())))
    
    if space_diff > tol or dir_diff > tol: return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dirs",  nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--json_list",  required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    k = len(args.fold_dirs)
    print(f"Ensembling {k} fold predictions → {args.output_dir}/")

    json_path = os.path.join(args.data_dir, args.json_list) \
        if not os.path.isabs(args.json_list) else args.json_list
    with open(json_path) as f:
        js = json.load(f)

    all_entries = []
    for split in ("validation", "training", "testing"):
        all_entries.extend(js.get(split, []))

    found = missing = 0
    for idx, entry in enumerate(all_entries):
        case_id = entry.get("case_id") or \
            os.path.basename(entry.get("npz", "unknown")).replace(".npz", "")

        output_path = os.path.join(args.output_dir, f"{case_id}_Pred.nii.gz")

        fold_preds: list = []
        for fold_dir in args.fold_dirs:
            cands = glob.glob(os.path.join(fold_dir, f"{case_id}_Pred.nii.gz"))
            if cands:
                fold_preds.append(sitk.ReadImage(cands[0]))
            else:
                print(f"  ⚠  [{case_id}] missing in {fold_dir}")

        if not fold_preds:
            print(f"  ❌ [{case_id}] No fold predictions found — skipping")
            missing += 1
            continue
        if len(fold_preds) < k:
            print(f"  ⚠  [{case_id}] Only {len(fold_preds)}/{k} folds available")

        reference = fold_preds[0]
        arrays    = [sitk.GetArrayFromImage(reference).astype(np.uint8)]
        for fp in fold_preds[1:]:
            if not check_geometry_match(fp, reference):
                fp = resample_to_reference(fp, reference)
            arrays.append(sitk.GetArrayFromImage(fp).astype(np.uint8))

        ensembled_arr  = majority_vote(arrays)
        ensembled_sitk = sitk.GetImageFromArray(ensembled_arr)
        ensembled_sitk.CopyInformation(reference)
        ensembled_sitk = sitk.Cast(ensembled_sitk, sitk.sitkUInt8)

        sitk.WriteImage(ensembled_sitk, output_path)
        found += 1
        print(f"  ✅ [{idx+1}/{len(all_entries)}] {case_id}  labels={np.unique(ensembled_arr).tolist()}")

    print(f"\nEnsemble complete: {found} saved, {missing} skipped")


if __name__ == "__main__":
    main()
