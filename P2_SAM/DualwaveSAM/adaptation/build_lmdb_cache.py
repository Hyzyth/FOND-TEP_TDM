# build_lmdb_cache.py
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import lmdb
import pickle
from dataset import extract_slices

def prebuild_lmdb(data_dir, json_list, lmdb_output_dir):
    data_dir = Path(data_dir)
    json_path = data_dir / json_list if not os.path.isabs(json_list) else Path(json_list)
    
    with open(json_path) as f:
        js = json.load(f)
        
    for split in ["training", "validation"]:
        entries = js.get(split, [])
        if not entries:
            continue
            
        print(f"\n📦 Compiling LMDB cache for [{split}] split...")
        json_stem = json_path.stem
        lmdb_path = Path(lmdb_output_dir) / f"{json_stem}_{split}.lmdb"
        
        # Open LMDB Environment
        env = lmdb.open(str(lmdb_path), map_size=1099511627776)
        has_fg_list = []
        global_idx = 0
        
        # 3. Stream data to disk patient-by-patient
        for entry in tqdm(entries, desc=f"Writing {split} to LMDB"):
            npz_rel = entry.get("npz", "")
            npz_path = str(data_dir / npz_rel)
            if not os.path.exists(npz_path):
                continue
                
            # Extract one patient
            ct_s, pet_s, lbl_s = extract_slices(npz_path)
            num_slices = ct_s.shape[0]
            
            # --- THE FIX: Open and close the transaction PER PATIENT ---
            with env.begin(write=True) as txn:
                for local_idx in range(num_slices):
                    # Grab exact slice
                    ct_slice  = ct_s[local_idx].astype(np.float32)
                    pet_slice = pet_s[local_idx].astype(np.float32)
                    lbl_slice = lbl_s[local_idx].astype(np.uint8)
                    
                    slice_record = {
                        "case_id": entry.get("case_id", "unknown"),
                        "ct": ct_slice,
                        "pet": pet_slice,
                        "label": lbl_slice
                    }
                    
                    # Track FG index instantly
                    has_fg = bool(np.any(lbl_slice > 0))
                    has_fg_list.append(has_fg)
                    
                    # Dump directly to hard drive cache
                    txn.put(f"{global_idx}".encode("ascii"), pickle.dumps(slice_record))
                    global_idx += 1
            # --- End of transaction block. RAM is flushed to disk here. ---

        # 4. Save the index block at the very end in a tiny final transaction
        with env.begin(write=True) as txn:
            txn.put(b"__fg_index__", pickle.dumps(has_fg_list))
            
        env.close()
        print(f"🔥 Successfully streamed {global_idx} slices to {lmdb_path}")

    print("\n✅ Setup Complete! All data compressed into blistering fast databases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/ethan/PP_hecktor2026_kfold_npz")
    parser.add_argument("--json_list", default="dataset_swincross_2026kfold_classic.json")
    parser.add_argument("--out_dir", default="/data/ethan/DualwaveSAM3c/lmdb_cache")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prebuild_lmdb(args.data_dir, args.json_list, args.out_dir)
