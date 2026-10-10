# SemEval 2027 Task 9, Subtask 2: project and Assignment 1 guide

This is the start-to-finish guide for **Reinforcement Learning (CS-317), Assignment 1, Group 12**. The repository contains the user-supplied first English Subtask 2 training chunk, a verified importer, data audit and EDA, two baseline families, saved held-out results, tests, and a LaTeX report. The code also supports the separate 20-image organizer trial. The attached 400-pair release is **chunk 1**, not proof that the complete multilingual competition data or official scorer have been released.

## 1. The actual task

SemEval 2027 Task 9 analyzes frames in news. **Subtask 2 predicts frames conveyed by an image** accompanying an article. The target is a subset of 14 generic frame labels: it may have multiple labels or none. The article is useful context but its labels are not the image's answer. The official primary measure is multilabel micro-F1.

| Configuration | Input | Prediction |
| --- | --- | --- |
| Image-only diagnostic | Image | Image frames |
| 2a | Image and article text | Image frames |
| 2b | Image, article text, supplied gold text-frame labels | Image frames |

The task covers Bulgarian, English, European Portuguese, and Russian. The **attached first training chunk has English only**. Its separate Subtask 1 corpus does not contain paired gold text-frame labels for the 400 Subtask 2 articles. Thus 2b cannot be run honestly on this release. The 20-image public trial has paragraph labels that we aggregate as a documented 2b proxy; those trial results are separate.

The response format is one TSV line per image: its exact organizer filename followed by zero or more tab-separated exact machine labels. The task description and public trial differ on the representation of an empty frame set; the local parser accepts both a filename-only row and a trailing tab. The local `--none-mode exclude` score counts frame predictions on an unframed image as false positives and adds no `None` class. **This score is provisional until the official scorer is checked.**

## 2. What data is included?

The tracked [`data/release1/data.zip`](data/release1/data.zip) is the exact user-uploaded archive, reassembled from three parts. It is 84,957,589 bytes with SHA-256 `ad18b9e877abb2f6a469bc7df0e855bebc684ead6a81d0e332765bff8e071fa9`; its ZIP integrity check passed. The `data/subtask2/` portion contains **400 labeled English image/article pairs**. Each row uses `EN_doc_N_image.jpg` with `EN_M_TRAIN_doc_N.txt`; the importer verifies the numeric join and all 400 image, article, and annotation IDs. Optional HTML files are viewing aids, not model input. The archive also contains separate Subtask 1 text datasets, which remain in the ZIP but are not treated as gold 2b labels for these images.

The tracked [`data/release1/manifest.csv`](data/release1/manifest.csv), [`framing-labels.txt`](data/release1/framing-labels.txt), and [`provenance.json`](data/release1/provenance.json) were generated from the archive. The manifest lists each organizer image identifier, local file path after extraction, article path, language, training split, exact image labels, and leakage group. There is **no** `text_labels` column because paired gold text frames are unavailable.

Verified audit of this chunk: all 400 pairs readable, no missing/corrupt files, no exact image/article duplicates or identical difference-hash image candidates, 72 unframed images, and 659 positive image-label assignments. Thirteen of 14 frame categories occur; `Legality_Constitutionality_and_jurisprudence` is absent. Four images carry 13 labels, an annotation pattern worth reviewing. Article lengths are 113–8,366 whitespace-separated words (median 514.5). These are observations about **this release**, not the eventual multilingual evaluation distribution.

## 3. Repository map

| Path | Role |
| --- | --- |
| `framing/release.py` | Validate and import the 400-pair attached training release. |
| `framing/trial.py` | Import the separate 20-image public organizer trial. |
| `framing/data.py` | Validate labels and manifest, resolve safe paths, load files. |
| `framing/audit.py` | Check data quality, compute descriptive features, create one EDA PDF. |
| `framing/features.py` | Extract 59 interpretable image statistics. |
| `framing/model.py` | Create grouped splits, fit baseline heads, calibrate, save metrics and predictions. |
| `framing/scoring.py` | Parse and locally score organizer-style TSV responses. |
| `framing/__main__.py` | CLI commands: `prepare-release`, `prepare-trial`, `audit`, `eda`, `train`, `predict`, `score`. |
| `tests/test_contract.py` | Import, manifest, split, empty-row, and scorer contract tests. |
| `artifacts/release1-*` | Portable audit, EDA PDF, split, scores, versions, and example predictions. |
| `report/main.tex`, `report/references.bib` | Assignment 1 LaTeX report and bibliography. |
| `submission/README.md`, `scripts/build_submission.py` | Upload package guide and reproducible ZIP builder. |
| `Agent.md`, `Changes.md`, `PLAN.md` | Working instructions, complete change log, and original plan. |

Raw extracted image/article files and run folders are regenerated locally and ignored by Git. The original ZIP and compact metadata are committed, so a clone has the entire supplied dataset.

## 4. Set up on your Mac from scratch

Use a working Python **3.11–3.13**; Python 3.12 is preferred. Python 3.14 previously failed while creating a virtual environment through `ensurepip`. Install Git and Python 3.12 first (`brew install python@3.12` is an option). For a new target folder:

```bash
git clone --branch work https://github.com/moizart/semeval-task9-2027.git "/Users/abdulmoiz/Documents/Semeval Task9"
cd "/Users/abdulmoiz/Documents/Semeval Task9"
bash scripts/setup-mac.sh
```

If that folder already contains this repository, run `cd "/Users/abdulmoiz/Documents/Semeval Task9"` and `bash scripts/setup-mac.sh`; the script fetches and fast-forwards `work`, selects a working Python, installs base dependencies in `.venv`, and runs contract tests. It refuses to overwrite an unrelated nonempty folder. The repository ZIP is about 85 MB, so clone/pull requires time and space. The setup script does not extract the dataset or download pretrained weights.

On any platform, manual setup is `python3.12 -m venv .venv`, then `./.venv/bin/python -m pip install -r requirements.txt`. Use `./.venv/bin/python` in every command below; activating `.venv` lets you shorten it to `python`.

## 5. Reproduce the attached-release workflow

From the repository root:

```bash
./.venv/bin/python -m framing prepare-release --archive data/release1/data.zip --out data/release1
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python -m framing audit --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/audit
./.venv/bin/python -m framing eda --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/eda
./.venv/bin/python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/classical --model classical --seed 42 --none-mode exclude
./.venv/bin/python -m pip install -r requirements-pretrained.txt
./.venv/bin/python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/frozen --model frozen --checkpoint torchvision:resnet18 --seed 42 --none-mode exclude
```

`prepare-release` extracts only the 400 Subtask 2 images and paired articles under `data/release1/extracted/`, validates labels and joins, and writes the manifest and provenance. It reads ZIP contents as data and does not run embedded files. Do **not** add `--enable-2b`: no paired gold text-frame input exists. The frozen run downloads torchvision's pretrained ResNet-18 weights once and caches them; it can take a few minutes on a CPU. On this runner, PyTorch's default home cache was read-only, so `TORCH_HOME=/tmp/semeval-torch` was set. A normal Mac home directory does not usually need that override.

To inspect results, open `runs/release1/audit/audit.json` for checks; `runs/release1/eda/eda.pdf` for the required single EDA plots PDF; and `runs/release1/{classical,frozen}/{image_only,2a_image_text}/metrics.json` and `evaluation_predictions.csv` for held-out scores, labels, probabilities and errors. Each family also saves `split_manifest.csv` and `versions.json`. The checked-in [`artifacts/README.md`](artifacts/README.md) explains the portable output snapshot.

## 6. How the models and evaluation work

The **classical baseline** measures image geometry, brightness, saturation, channel statistics, spatial brightness, edge density, and grayscale/color histograms (59 values). The **pretrained baseline** uses a frozen ImageNet ResNet-18 image encoder with a 512-value output. For 2a, each representation is joined with article word/bigram TF-IDF fitted only on the fit split. Fourteen independent class-weighted logistic heads predict the labels. A label absent in the fit subset gets a constant head, recorded in metrics; the absent legality frame cannot be learned from this release.

A deterministic grouped split with seed 42 produced 280 fit, 60 threshold-calibration and 60 held-out evaluation images. The split search uses all training labels to balance the partitions, including labels on the eventual evaluation rows. Scaling, TF-IDF and heads use fit only; calibration chooses a shared probability threshold; evaluation labels are then used for scoring. All four model configurations use the same split. The evaluation has 91 positive frame assignments and ten unframed images. This split is grouped by article ID, but source/template clusters remain to be checked when source metadata or later data become available.

| Model | Input | Local micro-F1 | Local macro-F1 | TP / FP / FN |
| --- | --- | ---: | ---: | --- |
| Classical | Image only | 0.272 | 0.218 | 50 / 227 / 41 |
| Classical | Image + article (2a) | 0.283 | 0.219 | 51 / 219 / 40 |
| Frozen ResNet-18 | Image only | 0.309 | 0.224 | 46 / 161 / 45 |
| Frozen ResNet-18 | Image + article (2a) | 0.308 | 0.223 | 46 / 162 / 45 |

These are **internal, provisional held-out scores**, not official competition results. The frozen model's selected 0.10 threshold is at the lower edge of the calibration search and produces many false positives. The small difference between 2a and image-only does not prove that text context is useless. The report analyzes representative errors. The saved heads are fit on only the 280 fit examples and are not final submission refits.

To check response formatting, `predict` writes one TSV row per image. For example, `./.venv/bin/python -m framing predict --run runs/release1/frozen/2a_image_text --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --split train --out runs/release1/format-check.tsv`. Because this command predicts on known training rows, use it only as a **format check**, never as held-out performance. To score a response against local gold, use `python -m framing score --gold path/to/gold.tsv --response path/to/response.tsv --labels data/release1/framing-labels.txt --none-mode exclude`. The scorer requires the exact same image-name set and valid labels in both files.

For a later organizer dataset, build a manifest with `image`, `image_path`, `article`, `language`, `split`, `image_labels`, and `group`. Use exact response identifiers in `image` and local paths relative to the manifest directory in `image_path`/`article`. Add `text_labels` only when the organizer supplies paired gold article frames; do not infer them from the target or unrelated text data. Audit every file and rerun models in a new output directory. Hidden dev/test labels must not guide thresholds or features.

## 7. Assignment 1 and what to submit

The supplied **Assignment 1: Shared Task / Challenge Project Guidelines** PDF asks for black-box and white-box task understanding; dataset size, labels, imbalance and quality checks; roughly 4–8 justified task-relevant EDA feature groups; and an **observation → interpretation → modeling implication** for patterns discussed. It requires **one PDF of relevant, self-explanatory EDA plots without traditional figure captions**, at least one interpretable classical baseline, one pretrained baseline, an explanation of the official metric, literature review, evaluation, error analysis, and a reproducible LaTeX report prepared using Overleaf. Its section list is: Introduction and Task Definition; Related Work; Dataset; Exploratory Data Analysis; Feature Engineering; Baseline Models; Evaluation Setup; Results; Error Analysis; Discussion; Next Steps.

| Assignment item | File/status |
| --- | --- |
| Source code and tests | `framing/`, `tests/`, `requirements*.txt`, `scripts/setup-mac.sh` |
| Supplied Subtask 2 training data | `data/release1/data.zip`, manifest, exact labels, provenance |
| One EDA plots PDF | `artifacts/release1-eda.pdf` |
| Audits, split, metrics, prediction evidence | `artifacts/release1-*` |
| Report source and references | `report/main.tex`, `report/references.bib`; editor compiler passed |
| Ready-to-upload ZIP with the exported report | Build with `python scripts/build_submission.py`; the ZIP appears under `dist/` |
| Official scorer/2b/later multilingual data | Not supplied; limitations stated in report |

The upload ZIP contains the PDF exported from Overleaf, the EDA PDF, the code, the supplied data and selected result evidence. Its README links to the user's Overleaf project. Repository-only planning notes, change logs and trial outputs are excluded. The guidelines do not state the required filename, upload portal, or deadline; confirm those with the instructor. The current report cover includes **Reinforcement Learning, CS-317, Assignment 1, Group 12** and all four supplied names/CMS numbers. The literature review discusses five related studies, cites the ResNet paper and links the organizer task page. An organizer baseline/system paper was not present in the supplied materials. Shared-task competition submission is separate: obtain its official starter kit/scorer and rules, refit after selection, and submit its required TSV only when the organizer opens that process.

## 8. Public trial, if you want the smaller demonstration

The public organizer [trial ZIP](https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip) has 20 labeled pairs, five per language. `prepare-trial` imports it, including paragraph frames as a **trial-only** 2b proxy. Results under `artifacts/trial-*` are kept for reproducibility but are superseded for Assignment 1 analysis by the 400-pair attached release. See `README.md` for trial commands. Its exact machine labels are in the ZIP's `labels.txt`; the separately linked task-page label file uses readable spelling and should not replace them.
