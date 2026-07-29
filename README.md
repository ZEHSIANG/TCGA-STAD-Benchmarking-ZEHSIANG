# Systematic Benchmarking of Foundation Models, MIL Architectures, and Multimodal Fusion for Gastric Cancer Survival Prediction on TCGA-STAD

**Tse-Hsiang Wang, Yen-Jung Chiu\***
Institute of Biomedical Engineering, Chang Gung University, Taoyuan, Taiwan
\*Corresponding author

> **Paper:** Submitted to ICBBE 2026 (ACM format, Ei/Scopus indexed)

---

## Overview

We benchmark **3 pathology foundation models × 3 MIL aggregators × 4 input modalities × 3 survival losses** for gastric cancer overall survival prediction on TCGA-STAD — **796 unique configurations, 3,980 training runs** — under a unified patient-level stratified 5-fold cross-validation protocol.

### Key Finding

> **Projection-based Fusion (SNN)** is the only design choice that significantly improves patient-level C-Index over WSI-only (0.660 vs. 0.640, Wilcoxon p=0.004, n=27 configurations). Foundation model and MIL architecture choices span gaps smaller than fold-level variability (gaps 0.006–0.012 vs. 5-fold SD 0.03–0.06) and are not separately distinguishable at this dataset scale.

---

## Pipeline

![Pipeline](figures/paper/fig01_pipeline.png)

*Figure 1. (A) Cohort curation and preprocessing pipeline. (B) Benchmarking design: full-factorial core grid, extension branches, and patient-level evaluation.*

---

## Dataset

![Dataset Statistics](figures/paper/fig02_dataset.png)

*Figure 2. TCGA-STAD cohort statistics: stage and gender distributions, survival status, OS time, age at diagnosis, WSIs per patient, slide-type composition, and per-fold event rate.*

| Property | Value |
|---|---|
| Cohort | TCGA-STAD |
| Patients (final analytic) | 381 |
| WSIs | 542 (367 TS + 175 BS) |
| Events (deaths) | 152 (39.9%) |
| Right-censored | 229 (60.1%) |
| Median OS | 467 days |
| Mean age at diagnosis | 65.1 years |
| AJCC Stage I / II / III / IV | 51 / 127 / 169 / 34 |
| Patients with 1 WSI | 220 (57.7%) |
| Patients with 2 WSIs | 161 (42.3%) |
| Cross-validation | Patient-level stratified 5-fold (seed = 42) |
| Raw cohort (before filtering) | 430 patients, 629 WSIs |

**Slide types:** TCGA-STAD contains only frozen sections — top section (TS) and bottom section (BS) are adjacent H&E sections cut from the upper and lower faces of the same tissue block. No FFPE diagnostic slides (barcode DX) are present in this cohort.

---

## Foundation Models

| Model | Tile size | Model input | Embedding dim | Token read-out | Pretraining data | Ref |
|---|---|---|---|---|---|---|
| **UNI2-h** | 256 px | 224 px | 1536 | CLS | >350,000 H&E/IHC WSIs (Mass General Brigham) | [1] |
| **Virchow2** | 224 px | 224 px | 2560 | CLS ⊕ mean patch | ~3.1M WSIs (diverse tissues/institutions, private) | [2] |
| **Midnight-12k** | 224 px | 224 px | 1536 | CLS | ~12,000 public TCGA WSIs | [3] |

**Important notes:**
- UNI2-h tiles at 256 px; each tile is resized to 224 px before the encoder (tile size ≠ model input size).
- Virchow2 uses CLS concatenated with mean patch token, as recommended by its authors.
- Midnight-12k uses CLS token only — departs from its own benchmark protocol (which recommends mean pooling).
- **Midnight-12k pretraining is entirely from TCGA**, overlapping our evaluation cohort (TCGA-STAD). Its relative performance should be interpreted with caution.
- UNI2-h was excluded from cross-FM feature fusion because its 256 px tiling grid is incompatible with the 224 px grid of the other two FMs.

---

## Methods

### MIL Architectures

| Architecture | Description |
|---|---|
| **MeanMIL** | Unweighted mean pooling (parameter-free baseline) |
| **ABMIL** | Gated attention-based aggregation [5] |
| **TransMIL** | Self-attention with Nyström approximation [6] |

### Fusion Strategies (both early fusion)

| Strategy | Description |
|---|---|
| **Concatenation Fusion** | WSI + clinical features concatenated directly before a shared prediction head |
| **Projection-based Fusion (SNN)** | Each modality projected to 256 dim via Self-Normalizing Network (Linear→ELU→AlphaDropout) before concatenation |

### Configuration Count

| Experiment | Unique configs | Training runs |
|---|---|---|
| Core full-factorial | 544 | 2,720 |
| Cross-FM feature fusion | 180 | 900 |
| Loss function comparison | 40 | 200 |
| Slide-inclusion ablation | 20 | 100 |
| Sample efficiency | 12 | 60 |
| **Total** | **796** | **3,980** |

---

## Results

### 1. Input Modality and Multimodal Fusion

![Input Modality](figures/paper/fig03_input_modality.png)

| Input modality | Mean C-Index | vs. WSI Only |
|---|---|---|
| Clinical Only | 0.6286 | — |
| WSI Only | 0.6405 | (reference) |
| Concatenation Fusion | 0.6446 | p=0.164 (n.s.) |
| **Projection-based Fusion (SNN)** | **0.6599** | **p=0.004** |

Projection-based Fusion also significantly beats Concatenation Fusion (p=0.008).

---

### 2. Foundation Model and MIL Architecture

![FM and MIL](figures/paper/fig04_fm_mil.png)

| Foundation model | Mean C-Index (WSI-only) | Δ vs. UNI2-h |
|---|---|---|
| UNI2-h | 0.6459 | (reference) |
| Virchow2 | 0.6417 | −0.004 |
| Midnight-12k | 0.6338 | −0.012 |

| MIL architecture | Mean C-Index (WSI-only) | Δ vs. ABMIL |
|---|---|---|
| ABMIL | 0.6431 | (reference) |
| TransMIL | 0.6416 | −0.002 |
| MeanMIL | 0.6366 | −0.006 |

> ⚠️ **Descriptive only.** With n=3 paired configurations, Wilcoxon p<0.05 is not attainable. Gaps of 0.006–0.012 are well within 5-fold SDs of 0.03–0.06.

---

### 3. Summary: All Design Dimensions

| Dimension | Level | C-Index | Paired comparison |
|---|---|---|---|
| Input modality (n=27) | Clinical Only | 0.6286 | — |
| | WSI Only | 0.6405 | (reference) |
| | Concatenation Fusion | 0.6446 | p=0.164 (n.s.) |
| | **Projection-based Fusion** | **0.6599** | **vs. WSI p=0.004; vs. Concat p=0.008** |
| Foundation model (n=3) | UNI2-h | 0.6459 | (reference) |
| | Virchow2 | 0.6417 | Δ=−0.004; not testable at n=3 |
| | Midnight-12k | 0.6338 | Δ=−0.012; not testable at n=3 |
| MIL architecture (n=3) | ABMIL | 0.6431 | (reference) |
| | TransMIL | 0.6416 | Δ=−0.002; not testable at n=3 |
| | MeanMIL | 0.6366 | Δ=−0.006; not testable at n=3 |
| Loss function (n=20) | Ranking Loss | 0.6527 | vs. Cox p=0.008 (Δ=+0.0043) |
| | Cox PH | 0.6483 | (reference) |
| | Logistic Hazard | 0.6323 | vs. Cox p=0.004 (Δ=−0.0160) |
| Slide inclusion (n=20) | All slides (TS+BS) | 0.6493 | (reference) |
| | TS Only | 0.6478 | p=0.452 (n.s.) |

---

### 4. Loss Function

![Loss](figures/paper/fig05_loss_comparison.png)

Ranking Loss is statistically superior (p=0.008) but practically equivalent (+0.004 C-Index). Logistic Hazard is clearly worse (Δ=−0.016, p=0.004).

---

### 5. Sample Efficiency

![Sample Efficiency](figures/paper/fig06_sample_efficiency.png)

| Training proportion | UNI2-h | Virchow2 | Midnight-12k |
|---|---|---|---|
| 25% (n≈76) | 0.5961 | 0.5930 | 0.5820 |
| 50% (n≈152) | 0.6111 | 0.6231 | 0.5927 |
| 75% (n≈228) | 0.6244 | 0.6292 | 0.5878 |
| 100% (n≈304) | 0.6452 | 0.6333 | 0.6312 |

UNI2-h and Virchow2 exchange rank at 50% and 75% with fully overlapping error bars — better described as noise than a consistent effect.

---

### 6. Top-10 Best Configurations (all 796)

| Rank | Configuration | Patient C-Index |
|---|---|---|
| 1 | Ranking Loss + Proj.Fusion + TransMIL + UNI2-h + p4000/bs32 | **0.6955 ± 0.0359** |
| 2 | Proj.Fusion + TransMIL + UNI2-h + p1000/bsFull | 0.6866 ± 0.0235 |
| 3 | Proj.Fusion + TransMIL + UNI2-h + p4000/bs32 + binary/one_hot | 0.6848 ± 0.0315 |
| 4 | Proj.Fusion + TransMIL + UNI2-h + p4000/bsFull | 0.6834 ± 0.0384 |
| 5 | Ranking Loss + Proj.Fusion + TransMIL + UNI2-h + p4000/bsFull | 0.6822 ± 0.0278 |
| 6 | Proj.Fusion + ABMIL + UNI2-h + p100/bsFull | 0.6796 ± 0.0295 |
| 7 | Proj.Fusion + TransMIL + Virchow2 + p100/bs32 | 0.6791 ± 0.0486 |
| 8 | Proj.Fusion + TransMIL + UNI2-h + p4000/bs32 | 0.6782 ± 0.0372 |
| 9 | Proj.Fusion + ABMIL + UNI2-h + p100/bs32 | 0.6779 ± 0.0428 |
| 10 | Ranking Loss + Proj.Fusion + ABMIL + UNI2-h + p100/bs32 | 0.6769 ± 0.0312 |

**All Top-10 use Projection-based Fusion (SNN). 9/10 use UNI2-h.**

---

### 7. Kaplan-Meier Survival Analysis

![KM](figures/paper/fig09_km_comparison.png)

Representative configuration (Proj.Fusion + TransMIL + UNI2-h, p=1000/bsFull): C-Index = 0.6866 ± 0.0235, log-rank **p<0.0001**.

Clinical-only baseline: non-significant under label encoding (p=0.194), significant under one-hot encoding (p=0.0001) — encoding choice materially affects extractable prognostic signal.

---

### 8. Attention Heatmaps

![Attention](figures/paper/fig10_attention.png)

![Attention FM Comparison](figures/paper/fig11_attention_fm.png)

> ⚠️ **Interpretation caveat:** These heatmaps show regions that the model assigns high attention weight to. **This study has not involved pathologist validation** — high-attention regions may or may not correspond to histologically meaningful cancer areas. The heatmaps are provided for qualitative exploration only. Future work should conduct systematic pathologist review to assess whether high-attention areas align with clinically relevant histological features.

---

## C-Index Bug Correction ⚠️

An early implementation computed patient-level C-Index using the **DataFrame row index** rather than actual OS time, inflating values by ~0.05 and distorting rankings. **All values in this repository are corrected.** Training, predictions, and KM curves are unaffected. Several previously "significant" differences (notably slide inclusion, previously p<0.001) did not survive correction.

---

## Limitations

1. **Frozen sections only** — TS/BS are frozen sections, not FFPE diagnostic slides used in most benchmarks
2. **Asymmetric read-out** — Virchow2 uses CLS+mean; Midnight-12k uses CLS only (departing from its protocol)
3. **Midnight-12k data overlap** — pretraining entirely from TCGA, overlapping evaluation cohort
4. **UNI2-h excluded from cross-FM fusion** — 256 px tiling incompatible with 224 px grid of other FMs
5. **Extension subsets pre-date correction** — selected on pre-correction rankings
6. **No external validation** — no other public gastric-cancer WSI+survival cohort available
7. **No pathologist validation of attention heatmaps**
8. **Statistical power** — FM/MIL comparisons rest on n=3 pairs; Wilcoxon p<0.05 not attainable

---

## Computational Environment

### Feature Extraction (5090 Server)
| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB) |
| CPU | AMD Threadripper PRO 9955WX |
| PyTorch | 2.13.0+cu130 |
| CUDA / cuDNN | 13.0 / 9.2.0 |
| timm | 1.0.26 |
| Feature extraction | TRIDENT framework |

### MIL Training (3060 Server)
| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3060 (12 GB) |
| CPU | Intel i5-13600KF |
| PyTorch | 1.12.0 |
| CUDA | 11.3 |
| lifelines | 0.30.0 |

---

## References

[1] Chen R.J. et al. Nat. Med. 30 (2024) 850–862.
[2] Zimmermann E. et al. arXiv:2408.00738.
[3] Karasikov M. et al. MICCAI 2025. arXiv:2504.05186.
[4] Bosch C. et al. J. Pathol. Inform. 21 (2026) 100648.
[5] Ilse M. et al. ICML 2018.
[6] Shao Z. et al. NeurIPS 2021.
[7] Cox D.R. J. R. Stat. Soc. Ser. B 34 (1972).
[8] Kvamme H., Borgan Ø. Lifetime Data Anal. 27 (2021).
[9] Steyaert S. et al. Nat. Mach. Intell. 5 (2023).
[10] Chen R.J. et al. ICCV 2021.
[11] TCGA. Nature 513 (2014).
[12] Zhu Z. et al. Eur. J. Surg. Oncol. 46 (2020).
[13] Harrell F.E. et al. JAMA 247 (1982).

---

## Funding

National Science and Technology Council, Taiwan (NSTC 113-2221-E-182-066-MY3)

---

## Citation

```bibtex
@inproceedings{wang2026stad,
  title     = {Systematic Benchmarking of Foundation Models, MIL Architectures,
               and Multimodal Fusion for Gastric Cancer Survival Prediction on TCGA-STAD},
  author    = {Wang, Tse-Hsiang and Chiu, Yen-Jung},
  booktitle = {Proceedings of ICBBE 2026},
  publisher = {ACM},
  year      = {2026}
}
```
