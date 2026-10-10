# SemEval 2027 Task 9: image framing baselines

Assignment 1 implementation for Subtask 2. The user-supplied **400-pair English training release (chunk 1)** is committed under `data/release1/`; the separate 20-image public trial remains available for a smaller demonstration. The original plan is in `PLAN.md`. Each row represents one **image** with its associated article. The image's zero or more framing labels are the target. Article framing labels, if supplied, are context for variant 2b only.

For a start-to-finish explanation of the task, code, local tests, outputs, and Assignment 1 deliverables, read [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Assignment upload package

[`submission/README.md`](submission/README.md) is the navigation page inside the upload package, including a placeholder for the Overleaf link. After editing that placeholder, run `python scripts/build_submission.py` to create `dist/Assignment_1_Group_12.zip` and a separate `dist/Assignment_1_Report_Group_12.pdf`. The builder needs `latexmk` and includes the attached data ZIP, code, saved results, LaTeX source, bibliography and compiled report. It checks the finished ZIP for damaged files.

## macOS setup

For a local Mac checkout, `scripts/setup-mac.sh` clones or updates the `work` branch, creates `.venv`, installs the base requirements, and runs the contract tests. Its default target is `/Users/abdulmoiz/Documents/Semeval Task9`. It selects a working Python 3.11–3.13 installation (preferring 3.12) and can rebuild a partial `.venv` left by a failed setup. It stops if the target contains unrelated files. The script does not download organizer data or a pretrained checkpoint.

## Run the attached 400-pair release

After setup, use the committed ZIP. Activate `.venv` or replace `python` below with `./.venv/bin/python`. The importer validates all 400 image/article/annotation IDs and extracts the necessary Subtask 2 files locally. The release has no paired gold text frames, so **do not use `--enable-2b`**:

```bash
python -m framing prepare-release --archive data/release1/data.zip --out data/release1
python -m framing audit --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/audit
python -m framing eda --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/eda
python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/classical --model classical --seed 42 --none-mode exclude
python -m pip install -r requirements-pretrained.txt
python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/frozen --model frozen --checkpoint torchvision:resnet18 --seed 42 --none-mode exclude
python -m unittest discover -s tests -v
```

On the fixed grouped 280/60/60 split, local held-out micro-F1 was 0.272/0.283 for classical image-only/2a and 0.309/0.308 for frozen image-only/2a. These are provisional local scores, not official organizer scores. The audit, EDA PDF, full split, metrics and selected predictions are checked into `artifacts/`; see `PROJECT_GUIDE.md` for interpretation and local Mac commands. The ZIP contains separate Subtask 1 text data, but no gold text-frame labels paired to these 400 Subtask 2 articles.

## Run the public organizer trial from scratch

The public [trial archive](https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip) has 20 labeled Subtask 2 images, five per language. It is a pipeline demonstration, **not** the full competition training data. From the repository root, after installing `requirements.txt`:

```bash
curl -fL 'https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip' -o data/trial.zip
python -m framing prepare-trial --archive data/trial.zip --out data
python -m framing audit --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/audit
python -m framing eda --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/eda
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/classical --model classical --enable-2b --seed 42
python -m pip install -r requirements-pretrained.txt
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/frozen --model frozen --checkpoint torchvision:resnet18 --enable-2b --seed 42
python -m unittest discover -s tests -v
```

The importer checks the ZIP's image-to-article pairs, copies its exact machine labels from `labels.txt`, combines paragraph text frames for 2b context, and writes `data/manifest.csv` plus `data/trial-provenance.json`. The ResNet-18 weights download from PyTorch on the first frozen run and are cached. Results appear in `runs/trial/`: `audit/audit.json`, `eda/eda.pdf`, and each model folder's `metrics.json` and `evaluation_predictions.csv`. A checked-in snapshot of the trial EDA and metrics is under `artifacts/`.

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

Rows with `split=dev` may leave `image_labels` blank; a blank train cell means an explicitly unframed image. Use the exact machine labels expected by the gold files. In the public trial archive these are in `labels.txt`; the separately linked task-page `framing-labels.txt` uses readable spelling and does not match the gold strings. The article/image mapping must be checked against the starter kit for any later dataset. The trial importer verifies matching stems for its own archive; do not assume that rule for the full training data without checking it. Set `image` to the exact submission identifier and `image_path` to its local path when it lives in a subdirectory.

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

`train` produces `image_only`, `2a_image_text`, and, with `--enable-2b` and verified text labels, `2b_image_text_gold` comparisons. The frozen model requires `requirements-pretrained.txt`. `torchvision:resnet18` uses frozen ImageNet-pretrained ResNet-18 weights and downloads them on first use. A locally available Hugging Face vision checkpoint is another option and is never downloaded implicitly. Only logistic heads are fit.

The internal split is grouped into fit, threshold-calibration, and held-out evaluation partitions. Thresholds are chosen on calibration only. Results are exploratory until the official scorer is run. In particular, the task page's `None` statement for Subtask 2 says “no framing in text”; `--none-mode` controls provisional local treatment (`exclude` or `include`). The default excludes unframed rows from true-positive accounting, while still counting predictions on them as false positives.

Generated run folders contain split snapshots, config, package versions, metric JSON, per-label scores, and predictions. Working extracts and generated runs stay ignored by Git. The user-supplied release ZIP and its compact metadata are tracked.

Build the draft report from its own directory so BibTeX finds `references.bib`:

```bash
cd report && latexmk -pdf main.tex
```

## Remaining inputs

The attached first English training chunk and the public trial have both been exercised. Later multilingual training chunks, gold article-frame context paired to Subtask 2 for 2b, the official starter kit, and the official scorer are not in this workspace. The report contains measured results for the attached 400-pair release. Verify the organizer scorer's empty-label behavior and final submission rules when available. The official site says detailed competition instructions will follow; confirm dates and allowed external resources there.
