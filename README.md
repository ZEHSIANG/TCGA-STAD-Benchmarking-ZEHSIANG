# Systematic Benchmarking of Pathology Foundation Models for Gastric Cancer Survival Prediction

**ICBBE 2025 | Chang Gung University | Advisor: Yen-Jung Chiu**

## Overview

This repository contains code and data splits for our systematic benchmarking study evaluating pathology foundation models (FMs) and multiple instance learning (MIL) architectures for gastric cancer overall survival prediction using TCGA-STAD.

> ⚠️ Note: Midnight-12k was pretrained on TCGA data. Results on TCGA-STAD should be interpreted with caution due to potential data overlap.

## Dataset

- Source: TCGA-STAD
- Patients: 381 | WSIs: 542 (TS=367, BS=175)
- Events (Dead): 152 (39.9%) | Censored: 229 (60.1%)
- OS Range: 14~3720 days

## Foundation Models

| Model | Dim | Patch Size | TCGA Pretrained |
|-------|-----|-----------|----------------|
| UNI2-h | 1536 | 256px | No |
| Virchow2 | 2560 | 224px | No |
| Midnight-12k | 1536 | 224px | Yes |

## Repository Structure
## Main Results (Patient-Level C-Index, 5-Fold CV)

| Input Mode | Foundation Model | MIL | C-Index |
|-----------|-----------------|-----|---------|
| Concat Fusion | Midnight-12k | MeanMIL | 0.6948 ± 0.0553 |
| Concat Fusion | Midnight-12k | ABMIL | 0.6948 ± 0.0700 |
| WSI Only | Midnight-12k | MeanMIL | 0.6797 ± 0.0492 |
| Clinical Only | - | - | 0.5707 ± 0.0483 |

Best configuration (binary task): 0.7021 ± 0.0387

## Environment

```bash
conda create -n STAD_benchmark python=3.9
conda activate STAD_benchmark
pip install torch lifelines scipy numpy pandas h5py openslide-python pycox
```

## Citation

```bibtex
@inproceedings{wang2025stad,
  title={Systematic Benchmarking of Pathology Foundation Models for Gastric Cancer Survival Prediction},
  author={Wang, Ze-Hsiang and Chiu, Yen-Jung},
  booktitle={ICBBE 2025},
  year={2025}
}
```
