# Bibliography

Reference papers for the FOND-TEP/TDM project. PDFs are stored under
`docs/SAM/`, `docs/SwinCross/` and `docs/Unused/`.

---

## Foundation model

- **SAM** — A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
  T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick.
  *Segment Anything.* ICCV 2023.
  arXiv: [2304.02643](https://arxiv.org/abs/2304.02643) ·
  Code: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## Models used in this project

- **SwinCross** (Phase 1) — G. Y. Li, J. Chen, S.-I. Jang, K. Gong, Q. Li.
  *SwinCross: Cross-modal Swin Transformer for Head-and-Neck Tumor Segmentation
  in PET/CT Images.* 2023.
  arXiv: [2302.03861](https://arxiv.org/abs/2302.03861) ·
  Medical Physics (2024), [doi:10.1002/mp.16703](https://doi.org/10.1002/mp.16703) ·
  Code: [yli192/SwinCross_CrossModalSwinTransformer…](https://github.com/yli192/SwinCross_CrossModalSwinTransformer_for_Medical_Image_Segmentation)

- **DualWaveSAM** (Phase 2) — J. Han, H. Chen, C. Li, C. Shen, Z. Li, H. Yao,
  W. Peng, J. Zhao, Q. Luo, X. Ding.
  *A PET/CT Cross-Modal Wavelet Fusion and Pseudo-Mask Guided Network with
  Frozen SAM Decoder for Multiple Myeloma Segmentation.* IEEE, 2026.
  [ieeexplore.ieee.org/document/11476882](https://ieeexplore.ieee.org/document/11476882) ·
  Code: [HanXinfun/DualwaveSAM](https://github.com/HanXinfun/DualwaveSAM)
  > Originally proposed for multiple-myeloma PET/CT segmentation; adapted here to
  > 3-class head-and-neck (GTVp / GTVn) segmentation.

- **MedSAM2** (Phase 2) — J. Ma et al.
  *MedSAM2: Segment Anything in 3D Medical Images and Videos.* 2025.
  arXiv: [2504.03600](https://arxiv.org/abs/2504.03600) ·
  Code: [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2)

---

## SAM variants surveyed but not retained

Reviewed during the literature survey and stored in `docs/Unused/`, but not used
in the benchmark (mostly single-modality and/or brain-MRI focused, i.e. domain
mismatch with 3D multimodal PET/CT H&N segmentation, or without usable public
code). Listed newest first.

- **Medical SAM3** — C. Jiang, T. Ding, C. Song, J. Tu, Z. Yan, Y. Shao,
  Z. Wang, Y. Shang, T. Han, Y. Tian.
  *Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image
  Segmentation.* arXiv, 2026. [arXiv:2601.10880](https://arxiv.org/abs/2601.10880)
- **Brain-SAM** — Y. Pan, X. Yuan, H. Liu, Y. Yang, G. Kang.
  *Brain-SAM: A SAM-based Model Tailored for Brain MRI Lesion Segmentation.*
  medRxiv, 2026.
  [medrxiv 2026.01.30.26345164v2](https://www.medrxiv.org/content/10.64898/2026.01.30.26345164v2.full.pdf)
- **SAM-RefiSeR** — D. Imans, P.-N. Bui, D.-T. Le, H. Choo.
  *Unsupervised Domain Adaptation with SAM-RefiSeR for Enhanced Brain Tumor
  Segmentation.* arXiv, 2026. [arXiv:2601.06882](https://arxiv.org/abs/2601.06882)
- **GBT-SAM** — C. Diana-Albelda, R. Alcover-Couso, Á. García-Martín, J. Bescós,
  M. Escudero-Viñolo.
  *GBT-SAM: A Parameter-Efficient Depth-Aware Model for Generalizable Brain
  Tumour Segmentation on mp-MRI.* arXiv, 2025.
  [arXiv:2503.04325](https://arxiv.org/abs/2503.04325)
- **MIQ-SAM3D** — J. Qu, J. Zhao.
  *MIQ-SAM3D: From Single-Point Prompt to Multi-Instance Segmentation via
  Competitive Query Refinement.* arXiv, 2025.
  [arXiv:2511.01345](https://arxiv.org/abs/2511.01345)
- **SMML** — G. Liang, Q. Zhou, Z. Wang, J. Chen, L. Gu, C. Yao, S. Wu, B. Huang,
  K. Chen.
  *Semantic-guided Masked Mutual Learning for Multi-modal Brain Tumor
  Segmentation with Arbitrary Missing Modalities.* AAAI, 2025.
  [ojs.aaai.org/…/32545](https://ojs.aaai.org/index.php/AAAI/article/view/32545)
- **Medical SAM-CLIP** — X. Yu, Z. Feng, X. Wu, J. Chen, W. Chen, B. Li,
  H. Kuang.
  *Medical SAM-CLIP Grafting for Brain Tumor Segmentation.* ScienceDirect, 2025.
  [S0010482525012806](https://www.sciencedirect.com/science/article/pii/S0010482525012806)
- **ESAM** — K. Ryu, Y. Jing, J. Zheng.
  *Enhancing Segment Anything Model (SAM) for Brain Tumor Image Segmentation.*
  Stanford CS231n Final Report, 2025.
  [cs231n.stanford.edu/2025](https://cs231n.stanford.edu/2025/papers/text_file_840592918-Final%20Report%20Submission.pdf)
- **Yolo-HLSAM** — B. Liu, H. Chen, T. Zhu, Z. Ye, H. Cui, K. Wang.
  *Yolo-HLSAM: Adapting Foundation Segment Anything Model for Semi-Automatic
  Detection and Segmentation of Breast Cancer Microcalcification Clusters.*
  ScienceDirect, 2025.
  [S1746809425008110](https://www.sciencedirect.com/science/article/pii/S1746809425008110)
- **Trans-SAM** — Y. Wu, Z. Wang, X. Yang, H. Kang, A. He, T. Li.
  *Trans-SAM: Transfer Segment Anything Model to Medical Image Segmentation with
  Parameter-Efficient Fine-Tuning.* ScienceDirect, 2025.
  [S0950705124015430](https://www.sciencedirect.com/science/article/pii/S0950705124015430)
- **RefSAM3D** — X. Gao, K. Lu.
  *RefSAM3D: Adapting SAM with Cross-modal Reference for 3D Medical Image
  Segmentation.* AccScience Publishing, 2025.
  [accscience.com/…/5442](https://accscience.com/journal/AIH/articles/online_first/5442)
- **DEAP-3DSAM** — F. Chen, J. Tang, P. Wang, T. Wang, S. Li, T. Deng.
  *DEAP-3DSAM: Decoder Enhanced and Auto Prompt SAM for 3D Medical Image
  Segmentation.* IEEE, 2024.
  [ieeexplore/10822764](https://ieeexplore.ieee.org/abstract/document/10822764)
- **SEG-SAM** — S. Huang, H. Liang, Q. Wang, C. Zhong, Z. Zhou, M. Shi.
  *SEG-SAM: Semantic-Guided SAM for Unified Medical Image Segmentation.* arXiv,
  2024. [arXiv:2412.12660](https://arxiv.org/abs/2412.12660)
- **LeSAM** — Y. Gu, Q. Wu, H. Tang, B. Li, H. Shu, X. Mai, Y. Chen.
  *LeSAM: Adapt Segment Anything Model for Medical Lesion Segmentation.* IEEE,
  2024. [ieeexplore/10540651](https://ieeexplore.ieee.org/document/10540651)
- **M-SAM** — H. Shi, S. Han, S. Huang, Y. Liao, G. Li, X. Kong, H. Zhu, X. Wang,
  S. Liu.
  *Mask-Enhanced Segment Anything Model for Tumor Lesion Semantic Segmentation.*
  arXiv, 2024. [arXiv:2403.05912](https://arxiv.org/abs/2403.05912)
- **SAM3D** — N.-T. Bui, D.-H. Hoang, M.-T. Tran, G. Doretto, D. Adjeroh,
  B. Patel, A. Choudhary, N. Le.
  *SAM3D: Segment Anything Model in Volumetric Medical Images.* arXiv, 2023.
  [arXiv:2309.03493](https://arxiv.org/abs/2309.03493)

---

## Datasets

- **HECKTOR 2025** — HEad and neCK TumOR Lesion Segmentation, Diagnosis and
  Prognosis. Grand Challenge. [hecktor25.grand-challenge.org](https://hecktor25.grand-challenge.org/)
- **HECKTOR 2026** — HEad and neCK TumOR Lesion Segmentation, Staging and
  Prognosis. Grand Challenge. [hecktor26.grand-challenge.org](https://hecktor26.grand-challenge.org/)
- **TEmPoRAL** — internal multicentre longitudinal PET/CT cohort,
  ClinicalTrials.gov [NCT02469922](https://clinicaltrials.gov/study/NCT02469922).
  Used **zero-shot** (evaluation only).
