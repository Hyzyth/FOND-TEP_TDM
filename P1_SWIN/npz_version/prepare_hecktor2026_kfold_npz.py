#!/usr/bin/env python3
"""
prepare_hecktor2026_kfold_npz.py
=================================
Offline preprocessing for SwinCross k-fold training using HECKTOR 2026 only.

Split logic
-----------
The 2026 dataset is split stratified by hospital acronym (prefix before '-'):

  X% of each acronym's patients  → training pool  (k-fold cross-validation)
  (1-X)% of each acronym's patients → fixed validation (never in any fold's training)

This keeps all hospital styles represented in both train and validation.

K-fold JSON generation
-----------------------
The training pool is stratified-split into k folds.  For fold i:
    training   = training_pool \\ fold_i
    validation = fold_i  ∪  fixed_validation

→  k JSONs : dataset_swincross_2026kfold_fold{i}.json
→  1 full  : dataset_swincross_2026kfold_full.json  (all pool, for final run)
→  split_info.json

Re-running is safe — existing NPZ/label files are skipped.

Usage
-----
  python npz_version/prepare_hecktor2026_kfold_npz.py \\
      --data_dir   "/data/santiago/HECKTOR_data/2026/HECKTOR 2026 Training Data" \\
      --output_dir /data/ethan/PP_hecktor2026_kfold_npz \\
      --train_ratio 0.8 \\
      --k_folds     5 \\
      --seed        42
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prepare_hecktor_npz_swincross import process_patient


# ── Dataset scanning ──────────────────────────────────────────────────────────

def get_acronym(folder_name: str) -> str:
    return folder_name.split("-")[0].upper()


def scan_dataset(data_root: Path) -> Dict[str, List[str]]:
    """Returns { acronym: [patient_id, ...] } for all patient folders."""
    acr_to_pids: Dict[str, List[str]] = defaultdict(list)
    for entry in sorted(data_root.iterdir()):
        if entry.is_dir() and "-" in entry.name:
            acr_to_pids[get_acronym(entry.name)].append(entry.name)
    return {k: sorted(v) for k, v in acr_to_pids.items()}


# ── Per-patient NPZ save ──────────────────────────────────────────────────────

def process_and_save(data_root: Path, pid: str,
                     npz_out: Path, lbl_out: Path) -> bool:
    if npz_out.exists() and lbl_out.exists():
        return True  # cached

    result = process_patient(data_root / pid, pid)
    if result is None:
        return False

    np.savez_compressed(
        str(npz_out),
        ct=result["ct"], pet=result["pet"], label=result["label"],
        ras_origin=result["ras_origin"], ras_direction=result["ras_direction"],
        ras_size_itk=result["ras_size_itk"],
        crop_start=result["crop_start"], crop_end=result["crop_end"],
        orig_spacing=result["orig_spacing"], orig_origin=result["orig_origin"],
        orig_direction=result["orig_direction"], orig_size_itk=result["orig_size_itk"],
    )
    sitk.WriteImage(result["_gt_sitk_orig"], str(lbl_out))
    return True


# ── K-fold JSON writer ────────────────────────────────────────────────────────

def write_kfold_jsons(train_entries: List[dict], val_entries: List[dict],
                      k: int, out_dir: Path, prefix: str, seed: int) -> List[str]:
    """
    Stratified-split train_entries into k folds (by source acronym),
    write k fold JSONs + 1 full JSON.
    """
    rng = random.Random(seed)

    # Group training entries by acronym for stratified splitting
    by_acr: Dict[str, List[dict]] = defaultdict(list)
    for e in train_entries:
        by_acr[e["acronym"]].append(e)

    # Build k stratified folds
    folds: List[List[dict]] = [[] for _ in range(k)]
    for acr, entries in by_acr.items():
        shuffled = list(entries)
        rng.shuffle(shuffled)
        for i, e in enumerate(shuffled):
            folds[i % k].append(e)

    print(f"\n{'─'*56}")
    print(f"K-Fold JSON generation  (k={k})")
    print(f"  Training pool : {len(train_entries)} cases")
    print(f"  Fixed val     : {len(val_entries)} cases")
    print(f"{'─'*56}")

    files = []
    for fi in range(k):
        val_fold  = folds[fi]
        train_set = [e for fj, fold in enumerate(folds) if fj != fi for e in fold]
        val_set   = val_fold + val_entries

        json_data = {
            "description": (
                f"HECKTOR 2026 — SwinCross {k}-fold CV, fold {fi} "
                f"| pool={len(train_entries)} fixed_val={len(val_entries)}"
            ),
            "labels": {"0": "background", "1": "GTVp", "2": "GTVn"},
            "training":   train_set,
            "validation": val_set,
        }
        fname = f"{prefix}_fold{fi}.json"
        with open(out_dir / fname, "w") as f:
            json.dump(json_data, f, indent=2)
        files.append(fname)
        print(f"  fold {fi}: train={len(train_set):4d}  val={len(val_set):4d} "
              f"(holdout={len(val_fold)}, fixed={len(val_entries)})")

    # Full-training JSON
    full_json = {
        "description": (
            f"HECKTOR 2026 — SwinCross full training "
            f"| pool={len(train_entries)} fixed_val={len(val_entries)}"
        ),
        "labels": {"0": "background", "1": "GTVp", "2": "GTVn"},
        "training":   train_entries,
        "validation": val_entries,
    }
    full_fname = f"{prefix}_full.json"
    with open(out_dir / full_fname, "w") as f:
        json.dump(full_json, f, indent=2)
    files.append(full_fname)
    print(f"  full:  train={len(train_entries):4d}  val={len(val_entries):4d}")

    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_dir)
    out_root  = Path(args.output_dir)
    rng       = random.Random(args.seed)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    # ── Scan dataset ──────────────────────────────────────────────────────
    print(f"Scanning {data_root} …")
    acr_to_pids = scan_dataset(data_root)
    total = sum(len(v) for v in acr_to_pids.values())
    print(f"  {len(acr_to_pids)} acronyms, {total} patients")
    for acr, pids in sorted(acr_to_pids.items()):
        print(f"    {acr}: {len(pids)} patients")

    # ── Stratified split by acronym ───────────────────────────────────────
    train_pids: List[Tuple[str, str]] = []   # (acronym, pid)
    val_pids:   List[Tuple[str, str]] = []

    for acr, pids in sorted(acr_to_pids.items()):
        shuffled = list(pids)
        rng.shuffle(shuffled)
        n_train = max(1, round(len(shuffled) * args.train_ratio))
        # Keep at least 1 case in val per acronym if possible
        if len(shuffled) > 1:
            n_train = min(n_train, len(shuffled) - 1)
        train_pids.extend((acr, p) for p in shuffled[:n_train])
        val_pids.extend((acr, p)   for p in shuffled[n_train:])

    print(f"\nSplit:  train={len(train_pids)}, val={len(val_pids)}")

    # ── Create output directories ─────────────────────────────────────────
    for sub in ["npz/train", "npz/val", "labels/train", "labels/val"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    skipped: List[str] = []

    # ── Preprocess training cases ─────────────────────────────────────────
    print(f"\nProcessing training cases…")
    train_entries: List[dict] = []
    for acr, pid in tqdm(sorted(train_pids), desc="train"):
        npz_rel  = f"npz/train/{pid}.npz"
        lbl_rel  = f"labels/train/{pid}_gt.nii.gz"
        ok = process_and_save(data_root, pid, out_root / npz_rel, out_root / lbl_rel)
        if not ok:
            skipped.append(pid); continue
        train_entries.append({"npz": npz_rel, "label": lbl_rel,
                               "case_id": pid, "acronym": acr})
    print(f"  → {len(train_entries)} processed")

    # ── Preprocess validation cases ───────────────────────────────────────
    print(f"\nProcessing validation cases…")
    val_entries: List[dict] = []
    for acr, pid in tqdm(sorted(val_pids), desc="val"):
        npz_rel  = f"npz/val/{pid}.npz"
        lbl_rel  = f"labels/val/{pid}_gt.nii.gz"
        ok = process_and_save(data_root, pid, out_root / npz_rel, out_root / lbl_rel)
        if not ok:
            skipped.append(pid); continue
        val_entries.append({"npz": npz_rel, "label": lbl_rel,
                             "case_id": pid, "acronym": acr})
    print(f"  → {len(val_entries)} processed")

    # ── Build JSONs ───────────────────────────────────────────────────────
    fold_files = write_kfold_jsons(
        train_entries=train_entries,
        val_entries=val_entries,
        k=args.k_folds,
        out_dir=out_root,
        prefix=args.json_prefix,
        seed=args.seed,
    )

    # ── Audit trail ───────────────────────────────────────────────────────
    split_info = {
        "parameters": {
            "data_dir": args.data_dir, "train_ratio": args.train_ratio,
            "k_folds": args.k_folds, "seed": args.seed,
        },
        "acronyms": sorted(acr_to_pids.keys()),
        "counts": {
            "train": len(train_entries), "val": len(val_entries),
            "skipped": len(skipped),
        },
        "fold_files": fold_files,
        "skipped": skipped,
    }
    with open(out_root / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\nDone.  Output → {out_root}/")
    if skipped:
        print(f"  ⚠  {len(skipped)} skipped — see split_info.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="HECKTOR 2026 → SwinCross NPZ with k-fold split")
    ap.add_argument("--data_dir",
                    default="/data/santiago/HECKTOR_data/2026/HECKTOR 2026 Training Data")
    ap.add_argument("--output_dir",
                    default="/data/ethan/PP_hecktor2026_kfold_npz")
    ap.add_argument("--train_ratio", type=float, default=0.8, metavar="X",
                    help="Fraction of each acronym's patients in the training pool. "
                         "Default: 0.8  (80%% train, 20%% fixed val).")
    ap.add_argument("--k_folds",    type=int,   default=5)
    ap.add_argument("--json_prefix", default="dataset_swincross_2026kfold")
    ap.add_argument("--seed",        type=int,  default=42)
    main(ap.parse_args())
