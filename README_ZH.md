# TidyLang2026：基于预训练模型与 Margin 损失的口语语种识别

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.01905-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2605.01905)

**[English](README.md) | [简体中文](README_ZH.md)**

</div>

## 任务

- **Task 1 — 已见语种识别（35 类）**：对每条语音做语种分类，指标为 macro / micro accuracy。
- **Task 2 — 未见语种验证**：仅基于注册语音与测试语音的相似度打分，指标为 EER。

## 方法概览

```
Audio (16 kHz)
      │
      ▼
ECAPA-TDNN (VoxLingua107 预训练)   ← 也可换为 XLS-R (facebook/wav2vec2-xls-r-300m)
      │
      ▼
Utterance Embedding
      │
      ▼
Margin Head:  AAM-Softmax  或  RAM-Softmax   ← 由 config.LOSS_TYPE 切换
      │
      ▼
Task 1: argmax 分类       Task 2: 取 embedding，cosine 相似度打分
```

- **编码器**：`speechbrain/lang-id-voxlingua107-ecapa`（ECAPA-TDNN）；备选 `facebook/wav2vec2-xls-r-300m`（XLS-R，接 Attentive Statistics Pooling）。
- **分类头**：AAM-Softmax（`ArcMarginProduct`）与 RAM-Softmax（`RAMSoftmax`），见 `slid_model.py`。

## 结果（论文 Table 1）

| System | Encoder | Loss | Macro Acc (%) ↑ | Micro Acc (%) ↑ | EER (%) ↓ |
|---|---|---|---|---|---|
| Baseline | Wav2Vec2-Large | AAM-Softmax | 40.25 | 75.76 | 34.70 |
| Ours | XLS-R | AAM-Softmax | 65.71 | 81.63 | — |
| Ours | ECAPA-TDNN | AAM-Softmax | 85.95 | 90.96 | 17.08 |
| Ours | ECAPA-TDNN | RAM-Softmax | 85.91 | 91.73 | 16.39 |

相较官方 baseline，macro accuracy 提升约 45.7，micro accuracy 提升约 15.2，EER 下降约 50.8%。

## 目录结构

```
release/
├── README.md                 # English
├── README_ZH.md              # 简体中文（本文件）
├── requirements.txt
├── config.py                 # 编码器/损失/超参/路径配置（运行前请按本机修改路径）
├── dataset.py                # 数据集与数据增强（MUSAN / RIRS）
├── slid_model.py             # 编码器 + AAM/RAM-Softmax 分类头
├── train.py                  # 训练 + checkpoint averaging
├── task1_identification.py   # Task 1：语种识别
├── task2_verification.py     # Task 2：未见语种验证（EER）
├── dummy_custom.py           # SpeechBrain 加载兼容占位文件
├── data/                     # 数据格式样例（不含真实音频）
│   ├── manifests/training_manifest.txt
│   └── trials/{enrollment_manifest.tsv, trials_Dev.txt}
└── eval_data/                # 评测列表格式说明
```

## 数据

实验基于 **Tidy-X** 数据集（由 Mozilla Common Voice 衍生）。请前往挑战赛官网获取数据，并在 `config.py` 中设置 `TRAIN_AUDIO_ROOT` / `VAL_AUDIO_ROOT` 等路径。数据增强需自行准备 MUSAN 与 RIRS 语料并设置 `MUSAN_ROOT` / `RIR_ROOT`。

`data/` 与 `eval_data/` 内仅提供**格式样例**，不含真实音频与完整列表，详见各目录下说明。

### Manifest 格式

`data/manifests/training_manifest.txt`，制表符分隔 `flag<TAB>相对路径<TAB>语种`：

```
1   id010001/en/en_30308892.wav   en
2   id010002/de/de_40923086.wav   de
3   id010003/fr/fr_50123456.wav   fr
```

| flag | 用途 |
|---|---|
| 1 | 训练集 |
| 2 | 验证集（新说话人，评分类准确率/语种识别） |
| 3 | 交叉语种验证集（已知说话人说不同语种） |

## 使用

运行前先在 `config.py` 中配置好编码器、`LOSS_TYPE`、数据路径。

```bash
# 训练（结束后自动做权重平均）
python train.py

# Task 1：语种识别，输出每条语音的预测语种
python task1_identification.py

# Task 2：未见语种验证，输出每个 trial 的相似度得分用于计算 EER
python task2_verification.py
```

## 引用

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

