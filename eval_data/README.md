# Evaluation list format (Task 1 / Task 2)

This directory contains **format samples only** — no real audio. Obtain the full
evaluation lists from the TidyLang Challenge 2026 website. Audio root dirs are set
via `TASK1_DATA_PATH` / `TASK2_*_PATH` in `config.py`.

## Task 1 — tl26_lid.txt

One audio file name per line. `task1_identification.py` predicts the language for
each line and writes the results in the same order.

## Task 2 — tl26_enroll.tsv

One enrollment ID per line: `enroll_id<TAB>wav1<TAB>wav2<TAB>...`, listing all
utterances of that enrollment ID.

## Task 2 — tl26_pairs.txt

One trial per line: `enroll_id<TAB>test_wav`. `task2_verification.py` computes the
cosine similarity between the enrollment embedding and the test embedding for each
trial and writes one score per line (used to compute EER).

> Note: the real `tl26_pairs.txt` is large (~112 MB) and is not included in the
> repo; `tl26_pairs.sample.txt` here is a format sample only.
