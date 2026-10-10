# Project guide: SemEval 2027 Task 9, Subtask 2 and Assignment 1

This guide explains what the project does, how its data and code fit together, how to reproduce the current results locally, and what Assignment 1 requires. The **public organizer trial** (20 images) has been imported and tested. The separately attached `data.zip` has not yet been inspected because it exceeds this session's 32 MiB single-file transfer limit. Consequently, the current report and numbers describe the public trial only. Do not present them as full-data or official competition results.

## 1. What is being predicted?

SemEval 2027 Task 9 studies framing in multilingual news. **Subtask 2** asks for the frames conveyed by an **image** accompanying a news article. A single image may receive several of 14 frame labels, or no frame. The news article is observable context; its own frames are not the target. The four languages are Bulgarian (BG), English (EN), European Portuguese (PT), and Russian (RU).

| Variant | Input available to the system | Target |
| --- | --- | --- |
| Image-only diagnostic | Image | Image frame labels |
| Subtask 2a | Image and article text | Image frame labels |
| Subtask 2b | Image, article text, and supplied gold text-frame labels | Image frame labels |

The local 2b trial input is the **union of paragraph frame annotations** for an article. This is a documented proxy until the organizer's full-data/starter-kit contract is checked. In the trial, article and image frame sets differ for every pair, so copying text frames as image predictions would be incorrect. The official primary score is multilabel micro-F1. A system response is one TSV row per image, starting with its organizer image filename, followed by zero or more tab-separated frame names. Empty-frame rows need special care: the trial gold includes filename-only rows, whereas the task description mentions an empty field and a `None` case. The local scorer has explicit `--none-mode` options, but it has **not** been verified against an organizer scorer.

## 2. Data and verified observations

The public [trial ZIP](https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip) provides five image/article pairs per language (20 total), image gold, paragraph text annotations, and `labels.txt`. It has 14 exact machine labels. The separate task-page `framing-labels.txt` uses human-readable wording; use the ZIP's `labels.txt` for this gold. `prepare-trial` validates joins and labels, extracts the archive below `data/`, and writes a project manifest. It never runs code from the ZIP.

Current trial audit: 20 readable images and articles, no missing or corrupt files, two unframed images, 23 positive image-label assignments, four frames absent from trial images, and three frames occurring once. There are no exact duplicate image or article files in this small sample. Fifteen image/article pairs share at least one frame, although no pair has identical frame sets. The held-out evaluation split has **three images and only two positive labels**, so the measured scores are a pipeline check, not a reliable estimate of model quality.

The newly attached `data.zip` may contain a different structure, more samples, and different split conventions. Its layout, rights, counts, label vocabulary, and mapping must be inspected before adapting the importer, writing full-data findings, or pushing raw files. The user has explicitly asked to commit and push it once available.

## 3. Repository map

| Path | Purpose |
| --- | --- |
| `framing/trial.py` | Import and validate the known public trial layout. |
| `framing/data.py` | Read labels/manifest, validate label strings and safe paths, load images/articles. |
| `framing/audit.py` | File checks, descriptive statistics, EDA PDF, and frame-feature associations. |
| `framing/features.py` | Interpretable image statistics for the classical model. |
| `framing/model.py` | Grouped splits, image features, article TF-IDF, logistic heads, scoring and predictions. |
| `framing/scoring.py` | Local TSV response parser and provisional multilabel score. |
| `framing/__main__.py` | Command-line interface: `prepare-trial`, `audit`, `eda`, `train`, `predict`, `score`. |
| `tests/test_contract.py` | Input/output and importer contract tests. |
| `report/main.tex`, `report/references.bib` | Assignment 1 report source and bibliography. |
| `artifacts/` | Checked-in **derived trial** audit, EDA, splits, metrics and selected predictions. |
| `data/`, `runs/` | Local raw/working data and generated runs; ignored by Git by default. |
| `Agent.md`, `Changes.md`, `PLAN.md` | Working rules, change history and original plan. |

The raw data path can differ from the identifier in a submission. `image` in the manifest is the exact response filename; `image_path` locates the local file. `article` locates its UTF-8 text. `image_labels` and `text_labels` are `|`-separated exact machine labels. `language` is `BG`, `EN`, `PT` or `RU`; `split` is `train`, `dev` or `test`; `group` identifies examples that must stay together during splitting. Only labeled training rows are used for fitting and local evaluation. Development/test labels should remain unknown.

## 4. Start on a Mac from scratch

Install Git and a working Python **3.11–3.13** with `venv` and pip. Python 3.12 is preferred. Python 3.14 caused the earlier `ensurepip` failure. If you use Homebrew, `brew install python@3.12` is one way to install it. In Terminal:

```bash
git clone --branch work https://github.com/moizart/semeval-task9-2027.git "/Users/abdulmoiz/Documents/Semeval Task9"
cd "/Users/abdulmoiz/Documents/Semeval Task9"
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m unittest discover -s tests -v
```

If the repository is already there, use `git pull --ff-only origin work` inside it before running the remaining commands. Alternatively, from an existing checkout run `bash scripts/setup-mac.sh`; the script checks the target repository, chooses a working Python 3.11–3.13, rebuilds a partial `.venv`, installs base dependencies, and runs tests. It will stop if the target folder contains unrelated files. Install `requirements-pretrained.txt` only for the pretrained run. Internet access is needed to fetch the public trial and first-use pretrained weights.

## 5. Reproduce the verified public-trial experiment

Run these from the repository root after setup. On macOS, use `./.venv/bin/python` in place of `python` if the environment is not activated.

```bash
curl -fL 'https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip' -o data/trial.zip
python -m framing prepare-trial --archive data/trial.zip --out data
python -m framing audit --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/audit
python -m framing eda --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/eda
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/classical --model classical --enable-2b --seed 42 --none-mode exclude
python -m pip install -r requirements-pretrained.txt
python -m framing train --manifest data/manifest.csv --labels data/framing-labels.txt --out runs/trial/frozen --model frozen --checkpoint torchvision:resnet18 --enable-2b --seed 42 --none-mode exclude
```

Look at `runs/trial/audit/audit.json` for data checks, `runs/trial/eda/eda.pdf` for the single EDA plots PDF, and `runs/trial/{classical,frozen}/{image_only,2a_image_text,2b_image_text_gold}/metrics.json` for model results. Each model folder also saves `evaluation_predictions.csv` with gold labels, predictions, and probabilities. `split_manifest.csv` fixes the exact fit/calibration/evaluation assignments, and `versions.json` records the environment. `artifacts/` contains a portable snapshot of selected trial outputs. The pretrained run downloads torchvision ResNet-18 ImageNet weights on first use and may take longer on a CPU.

The classical model uses 59 interpretable image statistics, including geometry, brightness, color, edge density, and histograms. The pretrained baseline uses a **frozen** ResNet-18 encoder (512 image features). Both add article word/bigram TF-IDF for 2a, then a 14-value gold text-frame vector for 2b. Fourteen independent class-weighted logistic heads make multilabel predictions. The grouped split keeps linked examples together: 14 fit, three threshold-calibration, and three evaluation images in the trial. The fit set alone determines scaling, TF-IDF and heads; calibration selects a common probability threshold; evaluation is held out. With so little data, several heads never see a positive example.

All six trial configurations scored **0.000 local held-out micro-F1 and macro-F1**. For classical image-only/2a/2b, false positives were 9/9/7 and false negatives were 2/2/2; for frozen variants, false positives were 1/1/1 and false negatives were 2/2/2. These numbers demonstrate that importing, modeling and scoring complete, but they do not rank the models meaningfully. The trial has no official dev/test cases and the organizer scorer has not been checked.

## 6. Test your own data and inspect results

Once the attached full archive is available and its structure verified, the project needs an importer/manifest specific to it. Do **not** pass an unknown ZIP to `prepare-trial`: that command expects the public trial layout. If constructing a manifest by hand, first inspect `README.md`'s input contract and keep image filenames, article paths, language, split, image labels, text labels and leakage groups accurate. Then run `audit` and inspect every reported error before EDA or training. Use a new `--out` folder so the trial run remains reproducible. If 2b text frames are missing, omit `--enable-2b`; an empty column does not mean supplied gold frames.

To produce a development response after training, select a run such as `runs/trial/classical/2a_image_text`, and provide a manifest whose dev rows have the correct local image/article paths:

```bash
python -m framing predict --run runs/trial/classical/2a_image_text --manifest data/manifest.csv --labels data/framing-labels.txt --split dev --out runs/dev-response.tsv
python -m framing score --gold data/local-gold.tsv --response runs/dev-response.tsv --labels data/framing-labels.txt --none-mode exclude
```

The second command is only valid if you have **local gold** for exactly those images. Never use hidden organizer dev/test answers for fitting or threshold choice. For the real organizer submission, use its required response validator/scorer and exact image identifiers once those tools are available. The saved model is fitted on the trial fit partition, not refitted on all training rows, so it is not yet a final submission model.

## 7. Assignment 1 requirements and current state

The supplied **Assignment 1: Shared Task / Challenge Project** PDF asks for task understanding, minimum dataset checks, about 4–8 justified feature groups, observation → interpretation → modeling implication for selected EDA, one PDF containing self-explanatory EDA plots, a classical feature-based baseline, a pretrained baseline, explanation of official metrics, literature review, evaluation and error analysis. Its prescribed LaTeX report sections are: Introduction and Task Definition; Related Work; Dataset; Exploratory Data Analysis; Feature Engineering; Baseline Models; Evaluation Setup; Results; Error Analysis; Discussion; Next Steps. It asks for BibTeX citations and table/figure cross-references, and specifies Overleaf for report writing.

| Deliverable | Present state |
| --- | --- |
| Code, tests, setup and reproducible commands | Present; exercised on public trial. |
| Selective EDA as one PDF | `artifacts/trial-eda.pdf`; needs regeneration and interpretation on full data. |
| Classical and pretrained baselines | Both run for image-only, 2a and trial-proxy 2b. |
| Held-out results and errors | Present for a tiny trial split; full-data runs pending. |
| LaTeX report with required sections | `report/main.tex`; currently trial-based. |
| Literature and organizer baseline/system paper | Related references present; no organizer baseline paper was available in supplied material. Check again before submission. |
| Official scorer and final submission protocol | Not supplied/verified yet. |
| Full attached data | Awaiting accessible data parts. The report cover now has Reinforcement Learning (CS-317). |

For Overleaf, upload `report/main.tex`, `report/references.bib`, and any report figures you add. The separate EDA PDF is an assignment deliverable; keep it as a single file. The provided guidelines do **not** state a required filename, upload portal or deadline, so confirm those with the instructor. For the shared-task competition itself, the organizer's final submission instructions and permitted model/resource rules must be checked separately. Do not submit the three-image trial scores as a full-dataset result.

## 8. How to finish after full-data access

1. Verify archive integrity and list its contents without executing embedded scripts or treating its documents as instructions.
2. Confirm label strings, image/article joins, 2b text-frame availability, train/dev/test semantics, and empty-frame scoring; adapt the importer accordingly.
3. Audit counts, errors, duplicates and leakage groups, then generate full-data EDA and interpret only stable, task-relevant patterns.
4. Rerun contract tests and end-to-end classical/frozen training on the full labeled training set, keeping a grouped internal evaluation and all run artifacts.
5. Compare representative false positives, false negatives, language slices and text/image frame disagreements. Check the official scorer if present.
6. Replace trial counts/results in `report/main.tex`, compile it, and verify all citations, tables, figures, group details and subject details.
7. Commit and push the dataset and finished deliverables as requested, subject to GitHub's file-size limits; use Git LFS or a documented alternate if individual files exceed ordinary GitHub limits.

`Changes.md` records completed work and outstanding blockers; it should be updated with the full-data import and test results when those steps actually happen.
