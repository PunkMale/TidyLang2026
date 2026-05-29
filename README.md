# TidyLang2026: Spoken Language Identification with Pre-trained Models and Margin Loss

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.01905-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2605.01905)

**[English](README.md) | [简体中文](README_ZH.md)**

</div>

## Tasks

- **Task 1 — Seen-language identification (35 classes)**: classify the language of each utterance. Metrics: macro / micro accuracy.
- **Task 2 — Unseen-language verification**: decide based only on the similarity between enrollment and test speech. Metric: EER.

## Method Overview

```
Audio (16 kHz)
      │
      ▼
ECAPA-TDNN (pre-trained on VoxLingua107)   ← or XLS-R (facebook/wav2vec2-xls-r-300m)
      │
      ▼
Utterance Embedding
      │
      ▼
Margin Head:  AAM-Softmax  or  RAM-Softmax   ← switch via config.LOSS_TYPE
      │
      ▼
Task 1: argmax classification       Task 2: cosine similarity scoring
```

- **Encoder**: `speechbrain/lang-id-voxlingua107-ecapa` (ECAPA-TDNN); alternative `facebook/wav2vec2-xls-r-300m` (XLS-R with Attentive Statistics Pooling).
- **Classification head**: AAM-Softmax (`ArcMarginProduct`) and RAM-Softmax (`RAMSoftmax`), see `slid_model.py`.

## Results (Table 1 of the paper)

| System | Encoder | Loss | Macro Acc (%) ↑ | Micro Acc (%) ↑ | EER (%) ↓ |
|---|---|---|---|---|---|
| Baseline | Wav2Vec2-Large | AAM-Softmax | 40.25 | 75.76 | 34.70 |
| Ours | XLS-R | AAM-Softmax | 65.71 | 81.63 | — |
| Ours | ECAPA-TDNN | AAM-Softmax | 85.95 | 90.96 | 17.08 |
| Ours | ECAPA-TDNN | RAM-Softmax | 85.91 | 91.73 | 16.39 |

Compared with the official baseline, macro accuracy improves by ~45.7, micro accuracy by ~15.2, and EER is reduced by ~50.8%.

## Directory Structure

```
release/
├── README.md                 # English (this file)
├── README_ZH.md              # 简体中文
├── requirements.txt
├── config.py                 # encoder / loss / hyper-params / paths (edit paths before running)
├── dataset.py                # dataset and data augmentation (MUSAN / RIRS)
├── slid_model.py             # encoder + AAM/RAM-Softmax head
├── train.py                  # training + checkpoint averaging
├── task1_identification.py   # Task 1: language identification
├── task2_verification.py     # Task 2: unseen-language verification (EER)
├── dummy_custom.py           # placeholder file for SpeechBrain loading
├── data/                     # data format samples (no real audio)
│   ├── manifests/training_manifest.txt
│   └── trials/{enrollment_manifest.tsv, trials_Dev.txt}
└── eval_data/                # evaluation list format reference
```

## Data

Experiments are based on the **Tidy-X** dataset (derived from Mozilla Common Voice). Obtain the data from the challenge website and set `TRAIN_AUDIO_ROOT` / `VAL_AUDIO_ROOT` in `config.py`. Data augmentation requires the MUSAN and RIRS corpora; set `MUSAN_ROOT` / `RIR_ROOT` accordingly.

`data/` and `eval_data/` contain **format samples only** — no real audio or full lists. See the README in each directory.

### Manifest format

`data/manifests/training_manifest.txt`, tab-separated `flag<TAB>rel_path<TAB>language`:

```
1   id010001/en/en_30308892.wav   en
2   id010002/de/de_40923086.wav   de
3   id010003/fr/fr_50123456.wav   fr
```

| flag | Purpose |
|---|---|
| 1 | Training set |
| 2 | Validation set (new speakers; classification accuracy / language recognition) |
| 3 | Cross-lingual validation (known speakers speaking a different language) |

## Usage

Configure the encoder, `LOSS_TYPE`, and data paths in `config.py` before running.

```bash
# Train (checkpoint averaging is triggered automatically at the end)
python train.py

# Task 1: language identification; outputs the predicted language per utterance
python task1_identification.py

# Task 2: unseen-language verification; outputs a similarity score per trial for EER
python task2_verification.py
```

## Citation

```bibtex
@misc{fang2026slid,
      title={Spoken Language Identification with Pre-trained Models and Margin Loss},
      author={Zhihua Fang and Liang He and Weiwu Jiang},
      year={2026},
      eprint={2605.01905},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.01905}
}
```

