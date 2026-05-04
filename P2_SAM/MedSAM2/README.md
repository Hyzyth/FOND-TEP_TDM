# HECKTOR MedSAM2

MedSAM2 fine-tuned for **HECKTOR Task-1**: automatic segmentation of primary
(GTVp) and nodal (GTVn) head-and-neck tumours from dual-modality **CT + PET**
volumes.

---

## Project structure

```
MedSAM2/
├── data_preparation/
│   ├── __init__.py
│   └── prepare_hecktor_npz.py    ← convert NIfTI → NPZ
│
├── sam2/                          ← core model (adapted from MedSAM2)
│   ├── __init__.py
│   ├── build_sam.py
│   ├── sam2_video_predictor_npz.py
│   ├── sam2_image_predictor.py
│   ├── configs/
│   │   ├── sam2.1_hiera_t512.yaml          ← base inference config
│   │   └── sam2.1_hiera_tiny_hecktor.yaml  ← HECKTOR fine-tuning config
│   ├── modeling/
│   │   ├── sam2_base.py
│   │   ├── sam2_utils.py
│   │   ├── memory_attention.py
│   │   ├── memory_encoder.py
│   │   ├── position_encoding.py
│   │   ├── backbones/
│   │   │   ├── hieradet.py
│   │   │   ├── image_encoder.py
│   │   │   └── utils.py
│   │   └── sam/
│   │       ├── mask_decoder.py
│   │       ├── prompt_encoder.py
│   │       └── transformer.py
│   ├── utils/
│   │   ├── amg.py
│   │   ├── misc.py
│   │   └── transforms.py
│   └── csrc/
│       └── connected_components.cu
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── loss_fns.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── sam2.py               ← SAM2Train
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── hecktor_dataset.py    ← NEW: CT+PET dual-modality loader
│   │   ├── vos_dataset.py
│   │   ├── vos_raw_dataset.py
│   │   ├── vos_sampler.py
│   │   ├── vos_segment_loader.py
│   │   ├── sam2_datasets.py
│   │   └── transforms.py
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py
│       ├── checkpoint_utils.py
│       ├── distributed.py
│       ├── logger.py
│       └── train_utils.py
│
├── inference/
│   ├── __init__.py
│   ├── infer_hecktor.py          ← NEW: full inference pipeline
│   └── evaluate_hecktor.py       ← NEW: DSC evaluation
│
├── scripts/
│   ├── download_checkpoints.sh
│   ├── train_hecktor.sh
│   └── infer_hecktor.sh
│
├── setup.py
└── README.md
```

---

## Label convention

| Value | Structure |
|-------|-----------|
| 0     | Background |
| 1     | GTVp (primary tumour) |
| 2     | GTVn (nodal tumour, may be absent) |

---

## Quick start

### 1 – Install

```bash
cd hecktor_medsam2
pip install -e ".[train]"
# Optional CUDA extension for hole-filling post-processing:
pip install -e ".[train]" --no-build-isolation
```

### 2 – Download checkpoints

```bash
bash scripts/download_checkpoints.sh
# Checkpoints saved to /data/ethan/MedSAM2/checkpoints/
```

### 3 – Prepare data

```bash
python data_preparation/prepare_hecktor_npz.py \
    --data_dir /data/santiago/HECKTOR_data/Task_1_segmentation \
    --output_dir /data/ethan/MedSAM2/hecktor_npz \
    --val_ratio 0.2 \
    --ct_low -200 --ct_high 800
```

This creates:
```
/data/ethan/MedSAM2/hecktor_npz/
    train/{patient_id}.npz
    val/{patient_id}.npz
    data_split.json
```

Each NPZ contains `ct_imgs`, `pet_imgs`, `gts`, `spacing`.

### 4 – Fine-tune

```bash
bash scripts/train_hecktor.sh
# or with 4 GPUs:
NUM_GPUS=4 bash scripts/train_hecktor.sh
```

Logs and checkpoints saved to `/data/ethan/MedSAM2/exp_log/hecktor_finetune/`.

### 5 – Run inference + evaluation

```bash
bash scripts/infer_hecktor.sh
# Override checkpoint:
CHECKPOINT=/data/ethan/MedSAM2/exp_log/hecktor_finetune/checkpoints/checkpoint_100.pt \
    bash scripts/infer_hecktor.sh
```

---

## Input modality fusion

CT and PET are fused into a 3-channel tensor `[CT, PET, PET]` before being
passed to the Hiera backbone. Both channels are normalised to `[0, 1]` then
standardised with ImageNet statistics `(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])`.

---

## Inference strategy

For each label (GTVp, GTVn) independently:
1. Find the **key slice** = axial slice with the largest annotation area.
2. Derive a **bounding-box prompt** from the GT mask on the key slice.
3. Run **forward propagation** (key slice → last slice).
4. Re-initialise and run **reverse propagation** (key slice → first slice).
5. Merge both passes via logical OR.

---

## Evaluation

Segmentation performance is measured by the **Dice Similarity Coefficient
(DSC)** for GTVp and GTVn separately; the final score is their mean.

```
DSC = 2·|P∩G| / (|P| + |G|)
```

Patients where GTVn is absent are excluded from the GTVn mean.

---

## Key files modified vs original MedSAM2

| File | Change |
|------|--------|
| `training/dataset/hecktor_dataset.py` | **NEW** – dual-modality CT+PET loader |
| `inference/infer_hecktor.py` | **NEW** – HECKTOR inference pipeline |
| `inference/evaluate_hecktor.py` | **NEW** – DSC evaluation |
| `data_preparation/prepare_hecktor_npz.py` | **NEW** – NIfTI → NPZ conversion |
| `sam2/configs/sam2.1_hiera_tiny_hecktor.yaml` | **NEW** – HECKTOR training config |
| `training/dataset/vos_raw_dataset.py` | Consolidated, removed dead branches |
| `training/dataset/vos_segment_loader.py` | Consolidated all loaders, added docstrings |
| `sam2/build_sam.py` | Updated docstrings, type hints |
| `setup.py` | Updated for HECKTOR project |
| All files | Uniform docstrings, import cleanup, checkpoint paths updated |
