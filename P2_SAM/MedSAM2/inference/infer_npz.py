"""
infer_npz.py
=============
MedSAM2 inference - supports two NPZ formats and three case-discovery modes.

NPZ formats (auto-detected per file)
-------------------------------------
SwinCross format  (HECKTOR 2026 training + test vault)
  Keys     : ct (R,A,S) int16 | pet (R,A,S) float16 | label (R,A,S) uint8
             + RAS inverse-transform metadata
  Slice axis: 2  (S = superior-inferior = axial, MONAI convention)
  Output   : NIfTI in original CT space via stored inverse transform
             -> compatible with SwinCross evaluate_predictions.py
  Raw SUV  : pet float16 values ARE raw SUV - passed directly to
             nestle/black/daisne without any uint8 round-trip

TemPoRAL format  (zero-shot, produced by prepare_temporal_npz.py)
  Keys     : ct_imgs (D,H,W) uint8 | pet_imgs (D,H,W) uint8
             gts (D,H,W) uint8 | spacing (sz,sy,sx) | pet_suv_max float32
  Slice axis: 0  (D = axial depth, SimpleITK GetArrayFromImage order)
  Output   : NIfTI written directly with stored spacing (already in patient space)
  Raw SUV  : reconstructed from uint8 via reconstruct_suv(pet_uint8, pet_suv_max)

Case-discovery modes
--------------------
SwinCross JSON  - --data_dir + --json_list (e.g. dataset_swincross_2026kfold_test.json)
  Reads "training" / "validation" / "testing" keys.

TemPoRAL manifest  - --data_dir + --json_list manifest.json
  Reads "cases" list (keys: case_id, npz_file, patient, timepoint, …).

Legacy directory  - --imgs_path
  Scans a plain directory for *.npz files.

Post-processing (all formats)
------------------------------
  A. Remove border-touching objects
  B. Remove GTVp shell around GTVn  (2D-slice-predictor artefact)
  C. Remove small connected components (GTVp < 100 mm³, GTVn < 50 mm³)
All steps logged to postprocessing_logs.csv (schema matches DualwaveSAM).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import OrderedDict
from glob import glob
from os.path import abspath, basename, dirname, join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage
from monai.transforms import RemoveSmallObjects
from skimage.segmentation import clear_border
from tqdm import tqdm

_REPO_ROOT = dirname(dirname(abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor_npz
from slicer import find_components, scale_bbox_2d, get_z_prompt_from_component

# ── Constants ──────────────────────────────────────────────────────────────────

IMG_MEAN       = (0.485, 0.456, 0.406)
IMG_STD        = (0.229, 0.224, 0.225)
MODEL_IMG_SIZE = 512
LABEL_GTVp     = 1
LABEL_GTVn     = 2

# Axial slice axis in MONAI (R,A,S) convention
_SWINCROSS_SLICE_AXIS = 2
# _TEMPORAL_SLICE_AXIS removed - TemPoRAL (D,H,W) already has axial on axis 0,
# no moveaxis needed unlike the SwinCross (R,A,S) path.

torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

CLASS_THRESHOLDS = {1: 100.0, 2: 50.0}   # mm³


# ── NPZ format detection ───────────────────────────────────────────────────────

def _detect_fmt(npz_path: str) -> str:
    """Return 'swincross' or 'temporal' from the NPZ keys present."""
    with np.load(npz_path, allow_pickle=False) as npz:
        keys = set(npz.files)
    if "ct" in keys and "pet" in keys and "label" in keys:
        return "swincross"
    if "ct_imgs" in keys and "pet_imgs" in keys:
        return "temporal"
    raise ValueError(
        f"Unknown NPZ format in {npz_path}. Keys: {keys}. "
        "Expected SwinCross keys (ct, pet, label) or "
        "TemPoRAL keys (ct_imgs, pet_imgs, gts)."
    )


# ── Intensity normalisation ────────────────────────────────────────────────────

def _norm_ct(ct: np.ndarray) -> np.ndarray:
    """int16 HU -> soft-tissue window [−160, +240] -> [0, 1] float32."""
    lo, hi = -160.0, 240.0
    return (np.clip(ct.astype(np.float32), lo, hi) - lo) / (hi - lo)


def _norm_pet(pet: np.ndarray) -> np.ndarray:
    """float16/float32 SUV -> 99th-percentile clip -> [0, 1] float32."""
    p = pet.astype(np.float32)
    p99 = float(np.percentile(p[p > 0], 99)) if (p > 0).any() else 1.0
    return np.clip(p / max(p99, 1e-6), 0.0, 1.0)


def _uint8_to_float(arr: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> [0, 1] float32."""
    return arr.astype(np.float32) / 255.0


# ── NPZ loading ────────────────────────────────────────────────────────────────

def _load_swincross(path: str) -> dict:
    """Load SwinCross NPZ.

    Returns
    -------
    ct_slices   (S, R, A) float32 [0,1] - for SAM2 backbone
    pet_slices  (S, R, A) float32 [0,1] - for SAM2 backbone
    pet_raw     (S, R, A) float32       - raw SUV for nestle/black/daisne
    lbl_slices  (S, R, A) uint8         - for GT prompts
    S, R, A     : int
    spacing_mm  : tuple (sx, sy, sz)  RAS isotropic spacing
    meta        : dict of inverse-transform arrays
    fmt         : 'swincross'
    """
    with np.load(path, allow_pickle=False) as npz:
        ct_ras  = npz["ct"].astype(np.float32)    # (R,A,S) int16 -> float32
        pet_ras = npz["pet"].astype(np.float32)   # (R,A,S) float16 -> float32  RAW SUV
        lbl_ras = npz["label"].astype(np.uint8)   # (R,A,S)
        meta    = {k: npz[k].copy() for k in npz.files
                   if k not in ("ct", "pet", "label")}

    # Keep raw SUV before normalisation (for auto-prompter)
    pet_raw = pet_ras.copy()

    # Normalise for SAM2 backbone
    ct_n  = _norm_ct(ct_ras)
    pet_n = _norm_pet(pet_ras)

    # (R,A,S) -> (S,R,A): axis 2 becomes axis 0
    ct_s  = np.moveaxis(ct_n,   _SWINCROSS_SLICE_AXIS, 0)
    pet_s = np.moveaxis(pet_n,  _SWINCROSS_SLICE_AXIS, 0)
    pet_r = np.moveaxis(pet_raw, _SWINCROSS_SLICE_AXIS, 0)
    lbl_s = np.moveaxis(lbl_ras, _SWINCROSS_SLICE_AXIS, 0)

    S, R, A = ct_s.shape

    # Derive RAS spacing for post-processing
    dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    dummy.SetSpacing([float(x) for x in meta["orig_spacing"]])
    dummy.SetDirection([float(x) for x in meta["orig_direction"].flatten()])
    spacing_mm = sitk.DICOMOrient(dummy, "RAS").GetSpacing()   # (sx,sy,sz)

    return dict(ct_slices=ct_s, pet_slices=pet_s, pet_raw=pet_r,
                lbl_slices=lbl_s, S=S, R=R, A=A,
                spacing_mm=spacing_mm, meta=meta, fmt="swincross")


def _load_temporal(path: str) -> dict:
    """Load TemPoRAL NPZ.

    Returns
    -------
    ct_slices   (D, H, W) float32 [0,1]
    pet_slices  (D, H, W) float32 [0,1]
    pet_raw     (D, H, W) float32       - reconstructed SUV (uint8 × suv_max / 255)
    lbl_slices  (D, H, W) uint8
    S=D, R=H, A=W : int
    spacing_mm  : tuple (sz, sy, sx)  - TemPoRAL native order (depth first)
    meta        : {'spacing': (sz,sy,sx), 'pet_suv_max': float}
    fmt         : 'temporal'
    """
    with np.load(path, allow_pickle=False) as npz:
        ct_raw  = npz["ct_imgs"]                        # (D,H,W) uint8
        pet_raw_u8 = npz["pet_imgs"]                    # (D,H,W) uint8
        lbl_raw = npz["gts"].astype(np.uint8)           # (D,H,W)
        spacing = npz["spacing"].copy()                  # (sz,sy,sx)
        suv_max = (float(npz["pet_suv_max"])
                   if "pet_suv_max" in npz.files else None)

    ct_f  = _uint8_to_float(ct_raw)
    pet_f = _uint8_to_float(pet_raw_u8)

    # Reconstruct raw SUV for auto-prompting thresholding methods
    if suv_max is not None and suv_max > 0:
        pet_raw_suv = pet_f * float(suv_max)   # ~ original SUV
    else:
        pet_raw_suv = None   # base41 still works; nestle/black/daisne will warn

    D, H, W = ct_f.shape
    meta = {
        "spacing": spacing, 
        "pet_suv_max": suv_max,
        "z_min": int(npz.get("z_min", 0)),
        "d_orig": int(npz.get("d_orig", D)),
        "origin": npz.get("origin"),
        "direction": npz.get("direction")
    }

    # spacing_mm in (sz,sy,sx) order - used directly for 3-D post-processing
    # because the array axes are (D,H,W) = (z,y,x) in sitk convention
    spacing_mm = tuple(float(s) for s in spacing)

    return dict(ct_slices=ct_f, pet_slices=pet_f, pet_raw=pet_raw_suv,
                lbl_slices=lbl_raw, S=D, R=H, A=W,
                spacing_mm=spacing_mm, meta=meta, fmt="temporal")


def _load_npz(path: str) -> dict:
    fmt = _detect_fmt(path)
    return _load_swincross(path) if fmt == "swincross" else _load_temporal(path)


# ── NIfTI export ───────────────────────────────────────────────────────────────

def _nifti_swincross(pred_sra: np.ndarray, meta: dict,
                     spacing_mm: tuple, out_path: str):
    """SwinCross: (S,R,A) pred -> (R,A,S) -> uncrop -> RAS -> original CT space."""
    pred_ras  = np.moveaxis(pred_sra, 0, _SWINCROSS_SLICE_AXIS)   # (R,A,S)
    ras_size  = [int(x) for x in meta["ras_size_itk"]]
    full      = np.zeros((ras_size[0], ras_size[1], ras_size[2]), dtype=np.uint8)
    cs, ce    = meta["crop_start"], meta["crop_end"]
    full[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = pred_ras

    # MONAI (R,A,S) -> ITK array (S,A,R)
    sitk_img = sitk.GetImageFromArray(full.transpose(2, 1, 0).astype(np.uint8))
    sitk_img.SetSpacing(spacing_mm)
    sitk_img.SetOrigin([float(x) for x in meta["ras_origin"]])
    sitk_img.SetDirection([float(x) for x in meta["ras_direction"].flatten()])

    orig_size = [int(x) for x in meta["orig_size_itk"]]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([float(x) for x in meta["orig_spacing"]])
    r.SetOutputOrigin([float(x) for x in meta["orig_origin"]])
    r.SetOutputDirection([float(x) for x in meta["orig_direction"].flatten()])
    r.SetSize(orig_size)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetOutputPixelType(sitk.sitkUInt8)
    sitk.WriteImage(r.Execute(sitk_img), out_path)


def _nifti_temporal(pred_dhw: np.ndarray, spacing_mm: tuple, out_path: str, meta: dict):
    sz, sy, sx = spacing_mm
    d_orig = meta.get("d_orig", pred_dhw.shape[0])
    z_min = meta.get("z_min", 0)

    # Uncrop along depth (Z) axis
    full = np.zeros((d_orig, pred_dhw.shape[1], pred_dhw.shape[2]), dtype=np.uint8)
    full[z_min:z_min+pred_dhw.shape[0]] = pred_dhw

    sitk_img = sitk.GetImageFromArray(full)
    sitk_img.SetSpacing((float(sx), float(sy), float(sz)))
    
    if meta.get("origin") is not None:
        sitk_img.SetOrigin([float(x) for x in meta["origin"]])
    if meta.get("direction") is not None:
        sitk_img.SetDirection([float(x) for x in meta["direction"]])

    sitk.WriteImage(sitk_img, out_path)


# ── Post-processing ────────────────────────────────────────────────────────────

def _remove_border(pred: np.ndarray, vox_vol: float):
    out   = pred.copy()
    bvols = {1: 0.0, 2: 0.0}
    bcnts = {1: 0,   2: 0}
    struct = ndimage.generate_binary_structure(3, 3)
    for cls in (1, 2):
        if not (pred == cls).any():
            continue
        mask    = (pred == cls)
        cleared = clear_border(mask)
        deleted = mask & (~cleared)
        out[deleted] = 0
        bvols[cls] = float(deleted.sum() * vox_vol)
        _, n = ndimage.label(deleted, structure=struct)
        bcnts[cls] = n
    return out, bvols[1], bvols[2], bcnts[1], bcnts[2]


def _remove_shell(pred: np.ndarray, vox_vol: float, iterations: int = 2):
    """Remove GTVp (label 1) pixels that form a shell around GTVn (label 2).

    This artefact occurs when the 2D slice predictor places tumour probability
    mass around the border of nodules.  A 3D dilation of GTVn followed by
    logical AND with GTVp identifies and removes the shell.
    """
    out    = pred.copy()
    n_mask = (out == 2)
    t_mask = (out == 1)
    if not n_mask.any() or not t_mask.any():
        return out, 0.0, 0
    struct   = ndimage.generate_binary_structure(3, 3)
    dilated  = ndimage.binary_dilation(n_mask, structure=struct, iterations=iterations)
    shell    = dilated & t_mask
    out[shell] = 0
    _, cnt = ndimage.label(shell, structure=struct)
    return out, float(shell.sum() * vox_vol), cnt


def _remove_small(pred: np.ndarray, vox_vol: float, thresholds: dict):
    out   = pred.copy()
    rvols = {1: 0.0, 2: 0.0}
    rcnts = {1: 0,   2: 0}
    struct = ndimage.generate_binary_structure(3, 3)
    for cls, thresh_mm3 in thresholds.items():
        if not (pred == cls).any():
            continue
        min_vox = max(1, math.ceil(thresh_mm3 / vox_vol))
        remover = RemoveSmallObjects(min_size=min_vox, connectivity=3)
        binary  = torch.from_numpy((pred == cls).astype(np.uint8)[None])
        filt    = remover(binary).numpy()[0]
        deleted = (out == cls) & (filt == 0)
        rvols[cls] = float(deleted.sum() * vox_vol)
        _, n = ndimage.label(deleted, structure=struct)
        rcnts[cls] = n
        out[deleted] = 0
    return out, rvols[1], rvols[2], rcnts[1], rcnts[2]


def _postprocess(pred: np.ndarray, spacing_mm: tuple):
    """Run all three stages; return cleaned prediction and tracking dict."""
    vox_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    pred, sv1, sc1               = _remove_shell(pred, vox_vol)
    pred, bv1, bv2, bc1, bc2     = _remove_border(pred, vox_vol)
    pred, smv1, smv2, smc1, smc2 = _remove_small(pred, vox_vol, CLASS_THRESHOLDS)
    return pred, {
        "bvol_p": bv1,  "bvol_n": bv2,
        "svol_p": sv1,
        "smvol_p": smv1, "smvol_n": smv2,
        "bcnt_p": bc1,  "bcnt_n": bc2,
        "scnt_p": sc1,
        "smcnt_p": smc1, "smcnt_n": smc2,
        "total_p": bc1 + sc1 + smc1,
        "total_n": bc2 + smc2,
    }


# ── SAM2 tensor fusion ─────────────────────────────────────────────────────────

def _to_tensor(ct_s: np.ndarray, pet_s: np.ndarray) -> torch.Tensor:
    """(S,H,W) float32 [0,1] CT+PET -> ImageNet-normalised (S,3,T,T)."""
    S = ct_s.shape[0]
    out = np.empty((S, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE), dtype=np.float32)
    for i in range(S):
        ct_r = np.array(
            Image.fromarray((ct_s[i]  * 255).astype(np.uint8))
                 .resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), Image.BILINEAR),
            dtype=np.float32) / 255.0
        pet_r = np.array(
            Image.fromarray((pet_s[i] * 255).astype(np.uint8))
                 .resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), Image.BILINEAR),
            dtype=np.float32) / 255.0
        out[i, 0] = ct_r
        out[i, 1] = pet_r
        out[i, 2] = pet_r
    t    = torch.from_numpy(out)
    mean = torch.tensor(IMG_MEAN)[:, None, None]
    std  = torch.tensor(IMG_STD)[:, None, None]
    return (t - mean) / std


# ── Overlay ────────────────────────────────────────────────────────────────────

def _save_overlay(ct_s: np.ndarray, segs: np.ndarray,
                  key_z: int, path: str):
    S = ct_s.shape[0]
    colors = {LABEL_GTVp: (1.0, 0.2, 0.2), LABEL_GTVn: (0.2, 0.6, 1.0)}
    idxs   = [max(0, S // 4), min(key_z, S - 1), min(S - 1, 3 * S // 4)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, idx, title in zip(axes, idxs, ["25th pct", "Key slice", "75th pct"]):
        ax.imshow(ct_s[idx], cmap="gray")
        for lid, col in colors.items():
            mask = (segs[idx] == lid).astype(np.float32)
            if not mask.any():
                continue
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[..., :3] = col
            rgba[..., 3]  = mask * 0.5
            ax.imshow(rgba)
        ax.set_title(f"{title} (s={idx})", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ── SAM2 propagation ───────────────────────────────────────────────────────────

@torch.inference_mode()
def _propagate(predictor, img_tensor, H, W, key_z, bbox, orig_hw, label_id, segs):
    box_m = scale_bbox_2d(bbox, orig_hw, MODEL_IMG_SIZE)

    def _write(fi, logits):
        mask = logits[0].squeeze(0).cpu().numpy() > 0
        segs[fi][mask & (segs[fi] == 0)] = label_id

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for reverse in (False, True):
            state = predictor.init_state(img_tensor, H, W)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=key_z, obj_id=1, box=box_m)
            for fi, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                _write(fi, logits)
            predictor.reset_state(state)


# ── Per-case inference ─────────────────────────────────────────────────────────

@torch.inference_mode()
def infer_one_case(npz_path: str, entry: dict, predictor,
                   pred_save_dir: str, overlay_dir, log_dir,
                   bbox_shift: int = 5, slice_pad: int = 1,
                   bbox_mode: str = "gt", auto_prompter=None):
    t0      = time.time()
    case_id = entry.get("case_id") or basename(npz_path).replace(".npz", "")
    print(f"\n> {case_id}")

    # ── Load (auto-detect format) ──────────────────────────────────────────
    d          = _load_npz(npz_path)
    fmt        = d["fmt"]
    ct_slices  = d["ct_slices"]    # (S,H,W) float32 [0,1]
    pet_slices = d["pet_slices"]   # (S,H,W) float32 [0,1]
    pet_raw    = d["pet_raw"]      # (S,H,W) float32 raw SUV  (or None for TemPoRAL without suv_max)
    lbl_slices = d["lbl_slices"]   # (S,H,W) uint8
    S, R, A    = d["S"], d["R"], d["A"]
    spacing_mm = d["spacing_mm"]   # tuple used for post-processing
    meta       = d["meta"]
    orig_hw    = (R, A)

    # ── SAM2 tensor ────────────────────────────────────────────────────────
    img_tensor = _to_tensor(ct_slices, pet_slices).cuda()

    # ── Prompts ────────────────────────────────────────────────────────────
    if bbox_mode == "gt":
        comps_per_label = find_components(
            mask        = lbl_slices,
            slice_pad   = slice_pad,
            planar_pad  = bbox_shift,
            label_values= (LABEL_GTVp, LABEL_GTVn),
        )
    else:
        if auto_prompter is None:
            raise RuntimeError(f"bbox_mode='{bbox_mode}' requires an AutoPrompter.")

        # pet_uint8 for base41 / UNet (scale-independent or internal normalisation)
        pet_u8 = (pet_slices * 255).astype(np.uint8)
        ct_u8  = (ct_slices  * 255).astype(np.uint8)

        # suv_max for TemPoRAL reconstruct_suv path
        suv_max = meta.get("pet_suv_max") if fmt == "temporal" else None

        # pet_raw for SwinCross direct-SUV path (None -> auto_prompter falls back)
        comps_per_label = auto_prompter.get_proposals(
            ct_uint8       = ct_u8,
            pet_uint8      = pet_u8,
            suv_max        = suv_max,        # TemPoRAL: float; SwinCross: None
            pet_suv_volume = pet_raw,        # SwinCross: float32 raw SUV; TemPoRAL: reconstructed float32
        )

    # ── Propagation ────────────────────────────────────────────────────────
    segs    = np.zeros((S, R, A), dtype=np.uint8)
    n_gtvp  = n_gtvn = 0

    if not comps_per_label:
        print("  [WARN] No foreground components - empty mask.")
    else:
        for label_id in sorted(comps_per_label):
            comps = comps_per_label[label_id]
            lname = "GTVp" if label_id == LABEL_GTVp else "GTVn"
            print(f"  {lname}: {len(comps)} component(s)")
            if label_id == LABEL_GTVp: n_gtvp = len(comps)
            else:                      n_gtvn = len(comps)
            for comp in comps:
                if bbox_mode == "gt":
                    key_z, bbox2d = get_z_prompt_from_component(comp)
                else:
                    key_z  = comp["z_mid"]
                    bbox2d = comp["bbox_2d"]
                print(f"    comp={comp['component_id']}  key_z={key_z}  "
                      f"bbox={bbox2d.tolist()}  voxels={comp['voxel_count']}")
                _propagate(predictor, img_tensor, R, A,
                           key_z, bbox2d, orig_hw, label_id, segs)

    # ── Post-processing ────────────────────────────────────────────────────
    segs, pp = _postprocess(segs, spacing_mm)
    duration = time.time() - t0

    # ── Save outputs ───────────────────────────────────────────────────────
    os.makedirs(pred_save_dir, exist_ok=True)
    nifti_path = join(pred_save_dir, f"{case_id}_Pred.nii.gz")

    if fmt == "swincross":
        _nifti_swincross(segs, meta, spacing_mm, nifti_path)
    else:
        _nifti_temporal(segs, spacing_mm, nifti_path, meta)
    print(f"  NIfTI: {nifti_path}")

    # NPZ pred for evaluate_npz.py
    np.savez_compressed(
        join(pred_save_dir, f"{case_id}.npz"),
        segs    = segs,
        spacing = np.array(spacing_mm),
        fmt     = fmt,
    )

    # Optional overlay
    if overlay_dir is not None:
        os.makedirs(overlay_dir, exist_ok=True)
        gtvp = comps_per_label.get(LABEL_GTVp, []) if comps_per_label else []
        kz   = (get_z_prompt_from_component(gtvp[0])[0]
                if gtvp and bbox_mode == "gt"
                else (gtvp[0]["z_mid"] if gtvp else S // 2))
        _save_overlay(ct_slices, segs, kz,
                      join(overlay_dir, f"{case_id}_overlay.png"))

    # Optional JSON timing log
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(join(log_dir, f"{case_id}.json"), "w") as f:
            json.dump({
                "case_id": case_id, "fmt": fmt,
                "bbox_mode": bbox_mode,
                "shape": list(segs.shape),
                "gtvp_components": n_gtvp,
                "gtvn_components": n_gtvn,
                "duration_s": round(duration, 2),
            }, f, indent=2)

    print(f"  Done in {duration:.1f}s")
    return case_id, duration, {"gtvp": n_gtvp, "gtvn": n_gtvn, **pp}


# ── Case discovery ─────────────────────────────────────────────────────────────

def _load_entries(args) -> list:
    """Return list of (abs_npz_path, entry_dict) from JSON, manifest, or directory."""
    if args.json_list:
        data_dir  = args.data_dir or "."
        json_path = (join(data_dir, args.json_list)
                     if not os.path.isabs(args.json_list) else args.json_list)
        with open(json_path) as f:
            js = json.load(f)

        # ── TemPoRAL manifest.json ─────────────────────────────────────
        if "cases" in js:
            return [
                (join(data_dir, e["npz_file"]), e)
                for e in js["cases"]
                if e.get("npz_file")
            ]

        # ── SwinCross JSON ─────────────────────────────────────────────
        entries = js.get(args.split, [])
        if not entries:
            for key in ("validation", "training", "testing"):
                entries = js.get(key, [])
                if entries:
                    break
        return [(join(data_dir, e["npz"]), e) for e in entries if e.get("npz")]

    # ── Legacy plain directory ─────────────────────────────────────────
    files = sorted(glob(join(args.imgs_path, "**", "*.npz"), recursive=True))
    files = [f for f in files if not basename(f).startswith("_")]
    return [(f, {"case_id": basename(f).replace(".npz", "")}) for f in files]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MedSAM2 inference (SwinCross NPZ + TemPoRAL NPZ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="./checkpoints/MedSAM2_latest.pt")
    p.add_argument("--cfg",        default="sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml")
    p.add_argument("--data_dir",   default=None,
                   help="NPZ root (SwinCross root OR TemPoRAL temporal_npz dir).")
    p.add_argument("--json_list",  default=None,
                   help="SwinCross JSON filename OR 'manifest.json' for TemPoRAL.")
    p.add_argument("--split",      default="validation",
                   help="JSON key for SwinCross JSONs (ignored for manifest.json).")
    p.add_argument("-i", "--imgs_path", default=None,
                   help="[Legacy] Plain NPZ directory.")
    p.add_argument("-o", "--pred_save_dir", required=True)
    p.add_argument("--bbox_shift",    type=int,   default=5)
    p.add_argument("--slice_pad",     type=int,   default=1)
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--bbox_mode",  default="gt",
                   choices=["gt", "pet", "unet", "hybrid"])
    p.add_argument("--pet_method", default="base41",
                   choices=["base41", "nestle", "black", "daisne"])
    p.add_argument("--proposal_model",  default=None)
    p.add_argument("--prob_threshold",  type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    if args.json_list is None and args.imgs_path is None:
        raise ValueError("Provide --data_dir + --json_list, or --imgs_path.")

    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)

    auto_prompter = None
    if args.bbox_mode != "gt":
        from auto_prompting import AutoPrompter
        auto_prompter = AutoPrompter(
            method        = args.bbox_mode,
            model_path    = args.proposal_model,
            pet_method    = args.pet_method,
            device        = "cuda" if torch.cuda.is_available() else "cpu",
            prob_threshold= args.prob_threshold,
            slice_pad     = args.slice_pad,
            planar_pad    = args.bbox_shift,
        )
        print(f"Auto-prompter: {auto_prompter}")

    entries = _load_entries(args)
    if not entries:
        raise RuntimeError("No cases found. Check --data_dir / --json_list.")
    print(f"Found {len(entries)} case(s). Format will be auto-detected per NPZ.")

    overlay_dir = join(args.pred_save_dir, "overlays") if args.save_overlays else None
    log_dir     = join(args.pred_save_dir, "logs")

    # ── Post-processing CSV ────────────────────────────────────────────────
    os.makedirs(args.pred_save_dir, exist_ok=True)
    csv_path = join(args.pred_save_dir, "postprocessing_logs.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "case_id", "patient", "timepoint", "study_date",
            "border_removed_GTVp_mm3", "border_removed_GTVn_mm3",
            "shell_removed_GTVp_mm3",
            "small_obj_removed_GTVp_mm3", "small_obj_removed_GTVn_mm3",
            "border_removed_GTVp_count", "border_removed_GTVn_count",
            "shell_removed_GTVp_count",
            "small_obj_removed_GTVp_count", "small_obj_removed_GTVn_count",
            "total_removed_GTVp_count", "total_removed_GTVn_count",
        ])

    timing  = OrderedDict()
    summary = []

    for npz_path, entry in tqdm(entries, desc="inference"):
        case_id = entry.get("case_id") or basename(npz_path).replace(".npz", "")

        if args.skip_existing:
            if os.path.exists(join(args.pred_save_dir, f"{case_id}_Pred.nii.gz")):
                print(f">>  Skip {case_id}")
                continue

        if not os.path.exists(npz_path):
            print(f"  ❌ Not found: {npz_path}")
            continue

        case_id_out, dur, pp = infer_one_case(
            npz_path      = npz_path,
            entry         = entry,
            predictor     = predictor,
            pred_save_dir = args.pred_save_dir,
            overlay_dir   = overlay_dir,
            log_dir       = log_dir,
            bbox_shift    = args.bbox_shift,
            slice_pad     = args.slice_pad,
            bbox_mode     = args.bbox_mode,
            auto_prompter = auto_prompter,
        )
        timing[case_id_out] = dur
        summary.append({"patient": case_id_out, "duration_s": f"{dur:.2f}",
                         "n_gtvp": pp["gtvp"], "n_gtvn": pp["gtvn"]})

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                case_id_out,
                entry.get("patient",    ""),
                entry.get("timepoint",  ""),
                entry.get("study_date", ""),
                round(pp["bvol_p"],  2), round(pp["bvol_n"],  2),
                round(pp["svol_p"],  2),
                round(pp["smvol_p"], 2), round(pp["smvol_n"], 2),
                pp["bcnt_p"], pp["bcnt_n"],
                pp["scnt_p"],
                pp["smcnt_p"], pp["smcnt_n"],
                pp["total_p"], pp["total_n"],
            ])

    with open(join(args.pred_save_dir, "inference_timing.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient", "duration_s"])
        for n, d in timing.items(): w.writerow([n, f"{d:.2f}"])

    with open(join(args.pred_save_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    n = len(timing)
    if n:
        total = sum(timing.values())
        print(f"\nAll done. {total:.1f}s total  ({total/n:.1f}s/case)")
    else:
        print("\nNo cases processed.")


if __name__ == "__main__":
    main()
