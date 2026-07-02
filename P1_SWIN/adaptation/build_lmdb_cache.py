# build_lmdb_cache.py
import os
import json
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import lmdb
import pickle

def build_3d_lmdb(data_dir, json_list, lmdb_output_dir):
    data_dir = Path(data_dir)
    json_path = data_dir / json_list if not os.path.isabs(json_list) else Path(json_list)
    
    with open(json_path) as f:
        js = json.load(f)
        
    for split in ["training", "validation"]:
        entries = js.get(split, [])
        if not entries:
            continue
            
        print(f"\n📦 Compiling 3D LMDB cache for [{split}] split...")
        json_stem = json_path.stem
        lmdb_path = Path(lmdb_output_dir) / f"{json_stem}_{split}.lmdb"
        
        env = lmdb.open(str(lmdb_path), map_size=1099511627776, writemap=True)
        global_idx = 0
        
        for entry in tqdm(entries, desc=f"Writing {split} to LMDB"):
            npz_rel = entry.get("npz", "")
            npz_path = str(data_dir / npz_rel)
            case_id = entry.get("case_id", f"unknown_{global_idx}")

            if not os.path.exists(npz_path):
                print(f"⚠️ Missing file: {npz_path}")
                continue
                
            with np.load(npz_path, allow_pickle=False) as npz:
                ct = npz["ct"]
                pet = npz["pet"]
                label = npz["label"]
                
                # Save just the lightweight metadata required for reconstruction
                meta = {
                    "case_id": case_id,
                    "npz_path": npz_path,
                    "ct_shape": ct.shape,
                    "ct_dtype": ct.dtype.str,
                    "pet_shape": pet.shape,
                    "pet_dtype": pet.dtype.str,
                    "label_shape": label.shape,
                    "label_dtype": label.dtype.str,
                }
            
                with env.begin(write=True) as txn:
                    # Store tiny metadata via pickle, but massive arrays as raw bytes
                    txn.put(f"{case_id}_meta".encode("ascii"), pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
                    txn.put(f"{case_id}_ct".encode("ascii"), ct.tobytes())
                    txn.put(f"{case_id}_pet".encode("ascii"), pet.tobytes())
                    txn.put(f"{case_id}_label".encode("ascii"), label.tobytes())
            
            env.sync()
            del ct, pet, label, meta
            gc.collect()
            global_idx += 1

        env.close()
        print(f"🔥 Successfully packed {global_idx} 3D volumes into {lmdb_path}")

    print("\n✅ SwinCross 3D LMDB Database build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack 3D SwinCross NPZ files into LMDB")
    parser.add_argument("--data_dir", default="/data/ethan/PP_hecktor2026_kfold_npz")
    parser.add_argument("--json_list", default="dataset_swincross_2026kfold_classic.json")
    parser.add_argument("--out_dir", default="/data/ethan/SwinCross_LMDB_cache")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    build_3d_lmdb(args.data_dir, args.json_list, args.out_dir)
