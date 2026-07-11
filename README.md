# StageM1_IA — FOND-TEP/TDM

**Foundation models for head-and-neck tumour segmentation in PET/CT.**

M1 research internship at the **LTSI** (Laboratoire Traitement du Signal et de
l'Image). This repository benchmarks three deep-learning segmentation models
against an **nnU-Net** baseline (developed separately by the supervising PhD
student) for automatic segmentation of primary (GTVp) and nodal (GTVn) tumours
from dual-modality **CT + PET** volumes.

The scientific context, objectives and hypotheses are described in the
internship report (`docs/`). In short: classical supervised architectures
(U-Net / nnU-Net, Vision Transformers) are compared against **SAM-based
foundation models**, and the transferability of these models is evaluated
zero-shot on an internal longitudinal cohort.

---

## The three studied models

The project is organised in two phases, mirroring the report:

| Phase | Model | Family | Role in the project | Folder |
|-------|-------|--------|---------------------|--------|
| 1 | **SwinCross** | Cross-modal Swin Transformer (supervised) | Transformer baseline with native CT+PET cross-modal attention | [`models/swincross/`](models/swincross/) |
| 2 | **DualWaveSAM** | SAM + wavelet encoder | SAM adaptation with a wavelet image encoder, retrained for 3-class output | [`models/dualwavesam/`](models/dualwavesam/) |
| 2 | **MedSAM2** | SAM 2 for 3D medical images/videos | Fine-tuned on HECKTOR + a novel automatic prompting pipeline | [`models/medsam2/`](models/medsam2/) |

Each model has its own README with setup, pipeline and references. The
**nnU-Net** baseline itself is **not** part of this repository.

---

## Datasets

| Dataset | Use | Notes |
|---------|-----|-------|
| **HECKTOR 2025 / 2026** | training + validation | Public challenge data (head-and-neck PET/CT). 2026 used with 5-fold cross-validation. |
| **TEmPoRAL** (NCT02469922) | **zero-shot evaluation only** | Internal multicentre longitudinal cohort (West of France, 2010–2017). Never used for training. |

Label convention (identical across all three models):

| Value | Structure |
|-------|-----------|
| 0 | Background |
| 1 | GTVp — primary tumour |
| 2 | GTVn — nodal tumour (may be absent) |

Evaluation metric: **Dice Similarity Coefficient (DSC)**, reported separately
for GTVp and GTVn (patients without GTVn are excluded from the GTVn mean).

---

## Repository layout

```
StageM1_IA/
├── README.md                  ← you are here
├── .gitignore
├── models/
│   ├── swincross/             ← Phase 1 — cross-modal Swin Transformer   (see its README)
│   ├── dualwavesam/           ← Phase 2 — SAM + wavelet encoder          (see its README)
│   └── medsam2/               ← Phase 2 — SAM 2, fine-tuned + auto-prompt (see its README)
├── utils/                     ← cross-model utilities (checkpoint trimming)
│   ├── trim_checkpoint.py
│   └── run_trimming.sh
└── docs/
    ├── BIBLIOGRAPHY.md        ← full paper list (used + surveyed-but-unused)
    ├── SwinCross/             ← reference papers (PDFs)
    ├── SAM/
    └── Unused/                ← papers surveyed but not retained
```

Inside each model folder, the same convention is used:

- `adaptation/` (or `data_preparation/` + `inference/` + `training/`) — **code
  written/modified during the internship** to plug the model into the
  HECKTOR / TEmPoRAL pipeline.
- `networks/`, `sam_modeling_wave/`, `sam2/` — **upstream model code**, kept as
  close to the original as possible (only minimal, documented changes).
- `*.sh` at the model root — the **shell entry points** that drive the whole
  pipeline (dataset building → training → inference → evaluation → plotting).
  Run them from the model folder (e.g. `cd models/swincross && bash SwinCross_NPZ_training.sh`).
- `requirements.txt` — per-model dependencies (each model uses its own venv).
- `obsolete/` — superseded code kept for reference only; **not maintained**,
  paths may be stale. Safe to ignore (and safe to delete — git history keeps it).
- `tools/` (SwinCross) — standalone inspection / debugging utilities.

> Each model is meant to run in **its own virtual environment**. Dependencies
> differ (MONAI/SimpleITK for SwinCross, the SAM stack for the two SAM models),
> so do not share a single env across models.

---

## Quick start

Pick a model and follow its README. The common pattern, driven by the scripts
in each `scripts/` folder, is:

```
1. build the NPZ dataset      (dataset-building script)
2. train / fine-tune          (training script)
3. run inference              (inference script)  → writes NIfTI predictions
4. evaluate                   (evaluation, DSC vs ground truth)
5. plot metrics / curves      (plotting script)
```

TEmPoRAL is only ever used at step 3–4 (zero-shot), never at step 2.

Paths to the data (`/data/santiago/HECKTOR_data/...`) and to outputs
(`/data/ethan/...`) are hard-coded as defaults in the scripts — **update them at
the top of each script** for a new environment/user.

---

## Models & primary references

| Model | Paper | Code |
|-------|-------|------|
| SAM (foundation) | Kirillov et al., *Segment Anything*, ICCV 2023 — [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| SwinCross | G. Y. Li, J. Chen, S.-I. Jang, K. Gong, Q. Li, *SwinCross: Cross-modal Swin Transformer for Head-and-Neck Tumor Segmentation in PET/CT Images*, 2023 — [arXiv:2302.03861](https://arxiv.org/abs/2302.03861) | [yli192/SwinCross_CrossModalSwinTransformer…](https://github.com/yli192/SwinCross_CrossModalSwinTransformer_for_Medical_Image_Segmentation) |
| DualWaveSAM | J. Han et al., *A PET/CT Cross-Modal Wavelet Fusion and Pseudo-Mask Guided Network with Frozen SAM Decoder for Multiple Myeloma Segmentation*, IEEE, 2026 — [ieeexplore/11476882](https://ieeexplore.ieee.org/document/11476882) | [HanXinfun/DualwaveSAM](https://github.com/HanXinfun/DualwaveSAM) |
| MedSAM2 | J. Ma et al., *MedSAM2: Segment Anything in 3D Medical Images and Videos*, 2025 — [arXiv:2504.03600](https://arxiv.org/abs/2504.03600) | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2) |

The complete bibliography — including the SAM variants that were **surveyed but
not retained** and the reason each was dropped — is in
[`docs/BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md).

---

## Notes for the next intern

- Start from this README, then read the target model's README end to end
  before running anything.
- The upstream folders (`networks/`, `sam_modeling_wave/`, `sam2/`) are kept
  faithful to the original repos; changes made for this project are documented
  at the top of each modified file and summarised in each model README.
- Large binaries (checkpoints `*.pt`/`*.pth`, generated `*.npz`, the compiled
  `_C.so`) are intentionally **git-ignored** — regenerate or download them via
  the scripts (`*_download_checkpoints.sh`, `*_dataset_building.sh`).
- The `docs/` PDFs are ~96 MB; if repository size becomes a problem, consider
  moving them to Git LFS or keeping only `BIBLIOGRAPHY.md` with links.

---

*LTSI — M1 internship, project FOND-TEP/TDM.*
