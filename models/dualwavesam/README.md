# DualWaveSAM — SAM + wavelet encoder (Phase 2)

Adaptation of **DualWaveSAM** (a SAM variant with a wavelet-based image encoder)
to **3-class HECKTOR** head-and-neck segmentation (background / GTVp / GTVn) on
**CT + PET**. This is one of the two Phase-2 SAM foundation models evaluated in
the project.

Upstream repo: [HanXinfun/DualwaveSAM](https://github.com/HanXinfun/DualwaveSAM)

DualWaveSAM operates on **2D axial slices** (not 3D patches), so batch sizes can
be large (48 for training, 32 for inference). Predictions are re-stacked into 3D
volumes for evaluation.

---

## What is upstream vs. adapted

| Folder / file | Origin | Content |
|---------------|--------|---------|
| `sam_modeling_wave/` | **upstream** | The SAM+wavelet model: `wave_encoder.py`, `image_encoder.py`, `prompt_encoder.py`, `mask_decoder.py`, `transformer.py`, `sam_model.py`, `common.py`. |
| `build_sam_wave.py`, `sam_wave.py` | **upstream** | Model builders / original entry modules. |
| `dataset/` | **upstream** | `dataSetGen.py`, `utils.py` — original 3D-NII → 2D-slice NPZ generation. |
| `adaptation/` | **written for this project** | 3-class wrapper (`model.py`), HECKTOR dataset (`dataset.py`), losses, LR scheduler, LMDB cache, training, inference, plotting. |
| `*.sh` (model root) | **written for this project** | Pipeline entry points. |
| `obsolete/` | archive | Superseded losses/metrics/predict/train. **Not maintained.** |

### The 3-class adaptation (`adaptation/model.py`)

The original DualWaveSAM forward pass is **preserved exactly**. The changes for
3-class output are deliberately thin (see the docstring at the top of
`model.py`):

- `MaskDecoder`: `num_multimask_outputs` 3 → 2 (giving 3 output channels total).
- `PseudoMaskHead`: output channels 1 → 3.
- A small `ClassAdaptor` (1×1 conv, 3→3) re-maps the raw mask channels to
  `(bg, GTVp, GTVn)` logits.

**Frozen** (as in the original): `PromptEncoder`, `MaskDecoder`.
**Trainable**: `WaveEncoder`, `PseudoMaskHead3`, `ClassAdaptor`.

Default model config: `IMG_SIZE=256`, `N_FILTERS=16`, `WAVELET=haar`,
`NUM_CLASSES=3`.

---

## Folder layout

```
dualwavesam/
├── adaptation/
│   ├── model.py                  ← 3-class wrapper around DualWaveSAM
│   ├── dataset.py                ← HECKTOR 2D-slice dataset
│   ├── build_lmdb_cache.py       ← NPZ → LMDB cache
│   ├── losses.py / lr_scheduler.py
│   ├── train.py / trainer.py     ← training entry + loop
│   ├── infer.py                  ← inference (+ post-processing)
│   ├── ensemble_kfold_predictions.py
│   └── plot_training.py / plot_postprocessing.py
├── sam_modeling_wave/            ← upstream model
├── dataset/                      ← upstream NII → 2D-slice NPZ generation
├── build_sam_wave.py / sam_wave.py
├── obsolete/                     ← archived (not maintained)
├── DualwaveSAM3c_build_cache.sh
├── DualwaveSAM3c_training.sh
├── DualwaveSAM3c_inference.sh
├── DualWaveSAM3c_temporal_zs.sh  ← zero-shot on TEmPoRAL
├── DualwaveSAM3c_run_plot.sh
└── requirements.txt
```

---

## Environment

```bash
uv venv dualwave_env --python 3.12
source dualwave_env/bin/activate
uv pip install -r requirements.txt
```

(The scripts create this venv automatically if missing.)

---

## Pipeline

Run scripts from this folder. Each has a config block at the top — set toggles
and update paths before running. The NPZ dataset is **shared with SwinCross**
(`/data/ethan/PP_hecktor2026_kfold_npz`), built by
`models/swincross/SwinCross_NPZ_Dataset_Building.sh`.

### 1 — Build the LMDB cache

```bash
bash DualwaveSAM3c_build_cache.sh
```

Toggles: `BUILD_CLASSIC`, `BUILD_KFOLD`, `BUILD_FULL`, `BUILD_EVAL`.

### 2 — Train

```bash
bash DualwaveSAM3c_training.sh
```

Modes: `RUN_CLASSIC_TRAIN`, `RUN_KFOLD_TRAIN`, `RUN_KFOLD_PRODUCTION_FULL`,
`RUN_CLASSIC_RESUME`. Defaults: `BATCH_SIZE=48`, `LRATE=1e-4`,
500 epochs (classic). Checkpoints under `runs/<model_dir>/`.

### 3 — Inference on HECKTOR

```bash
bash DualwaveSAM3c_inference.sh
```

`INFER_BATCH=32`. Evaluation and metric plots reuse the SwinCross scripts
(`../swincross/adaptation/evaluate_predictions.py`, `plot_metrics.py`) via the
`SWIN_DIR` variable at the top of the script.

### 4 — Zero-shot on TEmPoRAL

```bash
bash DualWaveSAM3c_temporal_zs.sh
```

Applies a trained model to the internal TEmPoRAL cohort **without any
fine-tuning**. Writes a prediction "vault", then evaluates against expert
delineations.

### 5 — Plot

```bash
bash DualwaveSAM3c_run_plot.sh
```

### Post-processing note

Inference includes post-processing of the raw SAM masks, notably removal of a
thin **GTVp "shell" around GTVn** predictions — a known artefact of the
2D-slice predictor. This is documented in the internship report ("zoom
technique") and logged to `postprocessing_logs.csv`.

---

## References

- **DualWaveSAM** — J. Han, H. Chen, C. Li, C. Shen, Z. Li, H. Yao, W. Peng,
  J. Zhao, Q. Luo, X. Ding. *A PET/CT Cross-Modal Wavelet Fusion and Pseudo-Mask
  Guided Network with Frozen SAM Decoder for Multiple Myeloma Segmentation.*
  IEEE, 2026.
  [ieeexplore.ieee.org/document/11476882](https://ieeexplore.ieee.org/document/11476882) ·
  Code: [github.com/HanXinfun/DualwaveSAM](https://github.com/HanXinfun/DualwaveSAM)
  *(originally for multiple-myeloma PET/CT; adapted here to 3-class H&N.)*
- **SAM** (foundation) — A. Kirillov et al., *Segment Anything*, ICCV 2023.
  arXiv: [2304.02643](https://arxiv.org/abs/2304.02643) ·
  Code: [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

See [`../../docs/BIBLIOGRAPHY.md`](../../docs/BIBLIOGRAPHY.md) for the full list.
