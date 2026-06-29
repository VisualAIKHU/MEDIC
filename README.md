# MEDIC

Official repository for **Different Changes Require Different Reasoning:
Change-Type-Specialized Experts for Robust Change Captioning**

**Jiyoung Park***, **InJae Oh***, and **Jung Uk Kim†**
Kyung Hee University
Accepted to ECCV 2026

* Equal contribution. † Corresponding author.

---

## Overview

**MEDIC** introduces change-type-aware reasoning for change captioning.
Instead of processing all visual changes with a single type-agnostic reasoning path, MEDIC routes each image pair through change-type-specialized experts.

MEDIC is designed as a plug-in module and can be integrated into existing change captioning models. In our paper, we apply MEDIC to representative baselines including DIRL, SMART, and SCORER.

Current release:

* Core MEDIC module
* DIRL + MEDIC training and evaluation code
* Configuration files for benchmark datasets
* Evaluation scripts

Additional integrations and checkpoints will be released soon.

---

## Architecture

<p align="center">
  <img src="assets/framework.png" width="900">
</p>

MEDIC consists of two main components:

1. **Two-stage router**
   The router first determines whether a meaningful change exists, and then predicts the corresponding change type.

2. **Change-type memory experts**
   Each expert is implemented as a key-value memory network and retrieves type-relevant visual patterns for more precise change reasoning.

---

## Installation

```bash
conda create -n medic python=3.8 -y
conda activate medic

pip install -r requirements.txt
```

---

## Data Preparation

Please prepare the datasets following the original benchmark settings.

Supported datasets:

* CLEVR-DC
* CLEVR-Change
* Spot-the-Diff
* Image Editing Request

Dataset preparation instructions will be provided in `docs/DATA_PREPARATION.md`.

---

## Training

Example command for training DIRL + MEDIC:

```bash
bash scripts/train_dirl_medic.sh
```

Detailed training instructions will be provided in `docs/TRAIN_AND_EVAL.md`.

---

## Evaluation

Example command for evaluating DIRL + MEDIC:

```bash
bash scripts/test_dirl_medic.sh
```

---

## Results

### Single-change setting

| Method       | Dataset       | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
| ------------ | ------------- | -----: | -----: | ------: | ----: | ----: |
| DIRL + MEDIC | CLEVR-DC      |      - |      - |       - |     - |     - |
| DIRL + MEDIC | CLEVR-Change  |      - |      - |       - |     - |     - |
| DIRL + MEDIC | Spot-the-Diff |      - |      - |       - |     - |     - |

The complete results and pretrained checkpoints will be released soon.

---

## Plug-in Usage

MEDIC can be integrated into existing change captioning models that produce paired visual difference features.

```python
# paired visual difference feature from a baseline model
paired_feature = torch.cat([diff_before, diff_after], dim=-1)

# MEDIC module
medic_feature, routing_info = medic(paired_feature)

# feed the enhanced representation to the caption decoder
decoder_input = torch.cat([paired_feature, medic_feature], dim=-1)
caption = decoder(decoder_input)
```

---

## Repository Structure

```text
MEDIC/
├── assets/
├── configs/
├── docs/
├── models/
│   └── medic/
├── scripts/
├── tools/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Citation

If you find this repository useful, please consider citing our paper.

```bibtex
@inproceedings{park2026medic,
  title={Different Changes Require Different Reasoning: Change-Type-Specialized Experts for Robust Change Captioning},
  author={Park, Jiyoung and Oh, InJae and Kim, Jung Uk},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```

---

## Acknowledgement

This repository is built upon publicly available change captioning codebases.
We sincerely thank the authors of DIRL, SMART, and SCORER for releasing their implementations.
