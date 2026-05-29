# Data format samples

This directory contains **format samples only** — no real audio. Obtain the real
data from the TidyLang Challenge 2026 website and set `TRAIN_AUDIO_ROOT` /
`VAL_AUDIO_ROOT` / `MANIFEST_PATH` in `config.py`.

## manifests/training_manifest.txt

Tab-separated, three columns: `flag<TAB>rel_path<TAB>language`.

| flag | Purpose |
|---|---|
| 1 | Training set |
| 2 | Validation set (new speakers) |
| 3 | Cross-lingual validation (known speakers speaking a different language) |

`rel_path` is probed under `TRAIN_AUDIO_ROOT` and `VAL_AUDIO_ROOT` in turn.

## trials/trials_Dev.txt

Trial list used to compute the language-recognition EER during training. One line
per trial: `label<TAB>enroll_id<TAB>test_wav`. `label`: 1 = target, 0 = non-target.

## trials/enrollment_manifest.tsv

One enrollment ID per line: `enroll_id<TAB>wav1<TAB>wav2<TAB>...`, listing all
enrollment utterances of that ID.
