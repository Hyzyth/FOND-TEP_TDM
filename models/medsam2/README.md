# MedSAM2 — SAM 2 for 3D medical images (Phase 2)

**MedSAM2** fine-tuned for **HECKTOR Task-1**: automatic segmentation of primary
(GTVp) and nodal (GTVn) head-and-neck tumours from dual-modality **CT + PET**
volumes. This is the second Phase-2 SAM foundation model, and the one for which
a **novel automatic-prompting pipeline** was developed so the model can run
**without any manual/ground-truth prompt**.

Upstream repo: [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)

---

## What is upstream vs. adapted

| Folder | Origin | Content |
|--------|--------|---------|
| `sam2/` | **upstream** | Core SAM 2 model (Hiera backbone, memory attention/encoder, mask decoder, predictors) + Hydra `configs/`. Minimal, documented edits. |
| `training/` | **upstream + adapted** | SAM 2 training stack; **new** `dataset/hecktor_dataset.py` (dual-modality CT+PET loader). |
| `data_preparation/` | **written for this project** | NIfTI → NPZ conversion for HECKTOR and TEmPoRAL. |
| `inference/` | **written for this project** | NPZ inference (`infer_npz.py`), DSC evaluation (`evaluate_npz.py`), metric/post-processing plots. |
| `auto_prompting/` | **written for this project (novel)** | Automatic bounding-box prompt generation (PET thresholding + learned proposal net). |
| `*.sh` (model root) | **written for this project** | Pipeline entry points. |
| `notebooks/` | upstream demo | `MedSAM2_inference_CT_Lesion.ipynb`. |

---

## Folder layout

```
medsam2/
├── data_preparation/
│   ├── prepare_hecktor_npz.py        ← HECKTOR NIfTI → NPZ (CT+PET+GT)
│   └── prepare_temporal_npz.py       ← TEmPoRAL → NPZ (zero-shot)
├── auto_prompting/                   ← NOVEL: automatic prompting
│   ├── auto_prompter.py              ← unified interface (pet | unet | hybrid)
│   ├── pet_proposal.py               ← PET thresholding (base41/nestle/black/daisne)
│   ├── proposal_net.py               ← Small3DUNet proposal network
│   ├── train_proposal_net.py         ← train the proposal net
│   └── box_utils.py
├── training/
│   ├── train.py / trainer.py / optimizer.py / loss_fns.py
│   ├── dataset/hecktor_dataset.py    ← NEW dual-modality CT+PET loader
│   └── model/sam2.py                 ← SAM2Train
├── inference/
│   ├── infer_npz.py                  ← full inference pipeline (SwinCross + TEmPoRAL NPZ)
│   ├── evaluate_npz.py               ← DSC evaluation
│   └── plot_metrics.py / plot_postprocessing.py
├── sam2/                             ← upstream SAM 2 core
│   └── configs/
│       ├── sam2.1_hiera_t512.yaml
│       ├── sam2.1_hiera_tiny_finetune512.yaml
│       ├── sam2.1_hiera_tiny_hecktor_finetune.yaml   ← HECKTOR fine-tuning
│       └── sam2.1_hiera_tiny_hecktor_infer.yaml      ← HECKTOR inference
├── notebooks/
├── MedSAM2_dataset_building.sh
├── MedSAM2_download_checkpoints.sh
├── MedSAM2_training.sh
├── MedSAM2_training_proposal_net.sh
├── MedSAM2_inference_execution.sh
├── MedSAM2_run_tests.sh
├── single_node_train_medsam2.sh
├── slicer.py / test_slicer.py / test_auto_prompting.py
├── setup.py
└── requirements.txt
```

---

## Install

```bash
uv venv medsam2_env --python 3.12   # or use the venv created by the scripts
source medsam2_env/bin/activate
pip install -e ".[train]"
# Optional CUDA extension for hole-filling post-processing:
pip install -e ".[train]" --no-build-isolation
```

---

## Input modality fusion

CT and PET are fused into a 3-channel tensor `[CT, PET, PET]` before the Hiera
backbone. Both channels are normalised to `[0, 1]`, then standardised with
ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## Label convention

| Value | Structure |
|-------|-----------|
| 0 | Background |
| 1 | GTVp (primary tumour) |
| 2 | GTVn (nodal tumour, may be absent) |

---

## Pipeline

Run scripts from this folder. Set toggles/paths in the config block at the top
of each.

### 1 — Download checkpoints

```bash
bash MedSAM2_download_checkpoints.sh
```

### 2 — Prepare data

```bash
bash MedSAM2_dataset_building.sh
```

The HECKTOR NPZ dataset is **shared with SwinCross**; if it does not exist yet,
build it first with `models/swincross/SwinCross_NPZ_Dataset_Building.sh`
(`BUILD_HECKTOR_2026_KFOLD=true`). Each NPZ contains `ct_imgs`, `pet_imgs`,
`gts`, `spacing`.

### 3 — Fine-tune

```bash
bash MedSAM2_training.sh          # or: NUM_GPUS=4 bash MedSAM2_training.sh
```

Uses `sam2/configs/sam2.1_hiera_tiny_hecktor_finetune.yaml`.

### 4 — (optional) Train the proposal network

```bash
bash MedSAM2_training_proposal_net.sh
```

Trains the `Small3DUNet` used by the `unet` / `hybrid` auto-prompting methods.

### 5 — Inference + evaluation

```bash
bash MedSAM2_inference_execution.sh
```

Uses `sam2.1_hiera_tiny_hecktor_infer.yaml`. Evaluation/metric plots reuse the
SwinCross scripts via `SWIN_DIR="../swincross"`.

### 6 — Tests

```bash
bash MedSAM2_run_tests.sh [ct|pet|both] [test_slicer|test_auto|all]
```

---

## Inference & auto-prompting

**Baseline inference strategy** (per label, GTVp / GTVn, independently):

1. Find the **key slice** (axial slice with the largest annotation area).
2. Derive a **bounding-box prompt** on the key slice.
3. **Forward-propagate** (key slice → last slice).
4. Re-initialise and **reverse-propagate** (key slice → first slice).
5. Merge both passes with a logical OR.

**Automatic prompting** (`auto_prompting/`) replaces the GT-derived box in
step 2 so the model can run with **no manual prompt**:

| `--pet_method` | Requires raw SUV | Idea |
|----------------|:---:|------|
| `base41` | no | 41 % SUVmax threshold (scale-invariant) |
| `nestle` | yes | `alpha·I0.7 + Ibgd` |
| `black`  | yes | iterative SUVmean |
| `daisne` | yes | iterative contrast |

Prompt sources (`--method`): `pet` (thresholding only), `unet` (learned
proposal net), `hybrid` = PET ∪ UNet filtered by 3-D IoU (recommended).

## Post-processing

Applied to all formats and logged to `postprocessing_logs.csv`:
(A) remove border-touching objects; (B) remove the GTVp shell around GTVn
(2D-slice-predictor artefact); (C) remove small connected components
(GTVp < 100 mm³, GTVn < 50 mm³).

## Evaluation

**Dice Similarity Coefficient (DSC)** for GTVp and GTVn separately; final score
is their mean. Patients without GTVn are excluded from the GTVn mean.

```
DSC = 2·|P∩G| / (|P| + |G|)
```

---

## References

- **MedSAM2** — J. Ma et al., *MedSAM2: Segment Anything in 3D Medical Images
  and Videos*, 2025.
  arXiv: [2504.03600](https://arxiv.org/abs/2504.03600) ·
  Code: [github.com/bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)
- **SAM** (foundation) — A. Kirillov et al., *Segment Anything*, ICCV 2023.
  arXiv: [2304.02643](https://arxiv.org/abs/2304.02643) ·
  Code: [github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

See [`../../docs/BIBLIOGRAPHY.md`](../../docs/BIBLIOGRAPHY.md) for the full list.
