# SemEval 2027 Task 9: image framing baselines

Assignment 1 implementation for Subtask 2. The original plan is in `PLAN.md`. Each row represents one **image** with its associated article. The image's zero or more framing labels are the target. Article framing labels, if supplied, are context for variant 2b only.

## macOS setup

For a local Mac checkout, `scripts/setup-mac.sh` clones or updates the `work` branch, creates `.venv`, installs the base requirements, and runs the contract tests. Its default target is `/Users/abdulmoiz/Documents/Semeval Task9`. It selects a working Python 3.11–3.13 installation (preferring 3.12) and can rebuild a partial `.venv` left by a failed setup. It stops if the target contains unrelated files. The script does not download organizer data or a pretrained checkpoint.

## Input contract

Create `data/manifest.csv` with these columns:

| Column | Meaning |
| --- | --- |
| `image` | Unique image filename exactly as the organizer expects in the response |
| `image_path` | Optional local path relative to `data/`; defaults to `image` |
| `article` | Article text path relative to `data/` |
| `language` | `BG`, `EN`, `PT`, or `RU` |
| `split` | `train` or `dev`; only train rows may have labels |
| `image_labels` | Exact labels separated by `|`; empty for a known unframed image |
| `text_labels` | Optional exact labels separated by `|` for 2b |
| `group` | Optional leakage group (article ID or image/template cluster); defaults to article path |
| `source` | Optional source identifier for EDA |

Rows with `split=dev` may leave `image_labels` blank; a blank train cell means an explicitly unframed image. Use the organizer's `framing-labels.txt` as `data/framing-labels.txt`, one exact label per line. The article/image mapping must be checked against the starter kit. The manifest is deliberately explicit because that mapping and the 2b annotation layout are not available yet. Do not infer a match from similar filenames without checking it. Set `image` to the exact submission identifier and `image_path` to its local path when it lives in a subdirectory.

## Commands

From the repository root, install `requirements.txt`, then:

```bash
python -m framing audit --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/audit
python -m framing eda --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/eda
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/classical --model classical --enable-2b
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/frozen --model frozen --checkpoint /path/to/local/vision-checkpoint
python -m framing predict --run runs/classical/2a_image_text --manifest data/manifest.csv --labels data/framing-labels.txt --split dev --out runs/dev-response.tsv
python -m framing score --gold data/local-gold.tsv --response runs/local-response.tsv --labels data/framing-labels.txt --none-mode exclude
```

`train` produces `image_only`, `2a_image_text`, and, when all training rows have text labels, `2b_image_text_gold` comparisons. The frozen model requires `requirements-pretrained.txt` and a locally available Hugging Face vision checkpoint (for example a CLIP or ViT image encoder). The checkpoint is frozen; only logistic heads are fit. The commands never download a checkpoint implicitly.

The internal split is grouped into fit, threshold-calibration, and held-out evaluation partitions. Thresholds are chosen on calibration only. Results are exploratory until the official scorer is run. In particular, the task page's `None` statement for Subtask 2 says “no framing in text”; `--none-mode` controls provisional local treatment (`exclude` or `include`). The default excludes unframed rows from true-positive accounting, while still counting predictions on them as false positives.

Generated run folders contain manifest and split snapshots, config, package versions, metric JSON, per-label scores, and predictions. Raw datasets and generated runs stay ignored by Git.

Build the draft report from its own directory so BibTeX finds `references.bib`:

```bash
cd report && latexmk -pdf main.tex
```

## Remaining inputs

The dataset ZIP, exact label list, starter kit/scorer, and a pretrained checkpoint are absent from this workspace. Once supplied, check the article mapping and scorer's empty-label behavior; then run the workflow and replace the pending tables and interpretations in `report/main.tex` with measured results. The official site says detailed submission instructions will follow. Confirm final dates and allowed external resources there before participation.
