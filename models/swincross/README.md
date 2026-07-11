# SwinCross — cross-modal Swin Transformer (Phase 1)

Adaptation of **SwinCross** (Cross-modal Swin Transformer) for **HECKTOR**
head-and-neck tumour segmentation from dual-modality **CT + PET** volumes.
This is the Phase-1 supervised transformer model, used as a strong
transformer reference alongside the nnU-Net baseline.

Upstream repo: [yli192/SwinCross_CrossModalSwinTransformer_for_Medical_Image_Segmentation](https://github.com/yli192/SwinCross_CrossModalSwinTransformer_for_Medical_Image_Segmentation)

---

## What is upstream vs. adapted

| Folder | Origin | Content |
|--------|--------|---------|
| `networks/` | **upstream** | SwinCross architecture (`SwinTransModels.py`), `unetr.py`, `configs_sw.py`. Minimal, documented edits only (e.g. `num_classes` for 3-class output). |
| `optimizers/` | **upstream** | `LinearWarmupCosineAnnealingLR`. |
| `adaptation/` | **written for this project** | Full HECKTOR/TEmPoRAL pipeline: NPZ preprocessing, MONAI dataloaders, LMDB cache, training, inference, evaluation, plotting. |
| `*.sh` (model root) | **written for this project** | Pipeline entry points. |
| `tools/` | **written for this project** | Inspection / debugging utilities (orientation checks, mask inspection, loader debugging). |
| `obsolete/` | archive | Superseded NIfTI-based pipeline and old test scripts. **Not maintained** (some imports still use the old `P1_SWIN.` package path). Kept for reference. |

The model outputs **3 classes** (0 = background, 1 = GTVp, 2 = GTVn). The main
upstream change is passing `num_classes = 3` through the config so the
segmentation head produces the expected number of labels.

---

## Folder layout

```
swincross/
├── adaptation/
│   ├── prepare_hecktor_npz_swincross.py    ← HECKTOR 2025 → NPZ
│   ├── prepare_hecktor2026_kfold_npz.py    ← HECKTOR 2026 → NPZ (5-fold)
│   ├── prepare_temporal_npz_swincross.py   ← TEmPoRAL → NPZ (zero-shot)
│   ├── build_lmdb_cache.py                 ← NPZ → LMDB cache (fast loading)
│   ├── data_utils.py                       ← MONAI dataloaders / transforms
│   ├── train.py  /  trainer.py             ← training entry + loop
│   ├── test.py                             ← inference → NIfTI in original CT space
│   ├── evaluate_predictions.py             ← DSC vs ground-truth NIfTI
│   ├── ensemble_kfold_predictions.py       ← merge k-fold predictions
│   ├── plot_metrics.py / plot_postprocessing.py
│   └── plot_training.py  (also at model root, called by run_plot.sh)
├── networks/            ← upstream architecture
├── optimizers/          ← upstream LR scheduler
├── tools/               ← inspection / debugging scripts
├── obsolete/            ← archived old pipeline (not maintained)
├── SwinCross_NPZ_Dataset_Building.sh
├── SwinCross_NPZ_training.sh
├── SwinCross_NPZ_inference_execution.sh
├── SwinCross_NPZ_run_plot.sh
└── requirements.txt
```

---

## Environment

Uses [`uv`](https://astral.sh/uv) for a reproducible Python 3.12 venv (the
scripts create it automatically). Manual setup:

```bash
uv venv swincross_env --python 3.12
source swincross_env/bin/activate
uv pip install -r requirements.txt
```

---

## Pipeline

All steps are driven by the four scripts. **Run them from this folder.** Each
script has a configuration block at the top — set the toggles and update the
data/output paths there before running.

### 1 — Build the NPZ dataset

```bash
bash SwinCross_NPZ_Dataset_Building.sh
```

Toggles inside: `BUILD_HECKTOR_2025`, `BUILD_HECKTOR_2026_KFOLD`,
`BUILD_TEMPORAL`. NPZ files store `ct` (int16, HU), `pet` (float16, SUV) and
`label` (uint8) plus RAS inverse-transform metadata, so predictions can be
written back into the original CT space. See the header of
`adaptation/prepare_hecktor_npz_swincross.py` for the exact NPZ schema.

### 2 — Train

```bash
bash SwinCross_NPZ_training.sh
```

Modes (toggles): `RUN_CLASSIC_TRAIN` (single split), `RUN_KFOLD_TRAIN`
(5-fold), `RUN_KFOLD_PRODUCTION_FULL` (final model on 100 % of the train pool),
`RUN_CLASSIC_RESUME` (resume from checkpoint). An **LMDB cache** is built first
for fast I/O. Checkpoints (`model_best.pth`, `model_last.pth`) are written under
`runs/<logdir>/`.

### 3 — Inference

```bash
bash SwinCross_NPZ_inference_execution.sh
```

Runs sliding-window inference (`INFER_OVERLAP=0.7` for quality, `0.5` for
speed) and writes NIfTI predictions in the original CT space. Independent
toggles for HECKTOR test/train/val and for **TEmPoRAL (zero-shot)**. For k-fold,
use `ensemble_kfold_predictions.py` to merge folds.

### 4 — Evaluate & plot

```bash
python adaptation/evaluate_predictions.py    # DSC (GTVp / GTVn) vs GT NIfTI
bash SwinCross_NPZ_run_plot.sh               # training curves
python adaptation/plot_metrics.py            # metric summaries
```

---

## References

- **SwinCross** — G. Y. Li, J. Chen, S.-I. Jang, K. Gong, Q. Li,
  *SwinCross: Cross-modal Swin Transformer for Head-and-Neck Tumor
  Segmentation in PET/CT Images*, 2023.
  arXiv: [2302.03861](https://arxiv.org/abs/2302.03861) ·
  Medical Physics (2024), [doi:10.1002/mp.16703](https://doi.org/10.1002/mp.16703) ·
  Code: [github.com/yli192/SwinCross…](https://github.com/yli192/SwinCross_CrossModalSwinTransformer_for_Medical_Image_Segmentation)
- **SAM** (foundation) — A. Kirillov et al., *Segment Anything*, ICCV 2023.
  arXiv: [2304.02643](https://arxiv.org/abs/2304.02643) ·
  Code: [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

See [`../../docs/BIBLIOGRAPHY.md`](../../docs/BIBLIOGRAPHY.md) for the full list.
