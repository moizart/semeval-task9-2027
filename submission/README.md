# Assignment 1 | Group 12

**Subject:** Reinforcement Learning (CS-317)

**Project:** SemEval 2027 Task 9, Subtask 2

**Overleaf report:** https://www.overleaf.com/read/tgmkhfcwvwvd#e4cd5d

**Group members:** Abdul Moiz (CMS 521568), Muhammad Ibsham Tariq (CMS 502451), Abdullah Khalil (CMS 501492) and Zia Ur Rehman (CMS 511386).

## Files in this submission

| Path | Purpose |
| --- | --- |
| `A1_Semeval_Task9_Group12_Report.pdf` | The report exported from Overleaf, including the literature review and references. |
| `artifacts/release1-eda.pdf` | The single PDF of selected EDA plots. |
| `artifacts/release1-metrics.csv` | Held-out scores for the four reported baselines. |
| `artifacts/release1-audit.json` | Data quality and label counts for the 400 pairs. |
| `artifacts/release1-split-manifest.csv` | The saved fit, calibration and evaluation split. |
| `data/release1/data.zip` | The supplied English Subtask 2 training chunk. |
| `framing/` | Data import, audit, feature extraction, modeling, prediction and scoring code. |
| `tests/test_contract.py` | Checks for data joins, splits, labels and response scoring. |
| `requirements.txt`, `requirements-pretrained.txt` | Python dependencies for the classical and frozen ResNet-18 runs. |

This is the first English training chunk: 400 image and article pairs. Its separate Subtask 1 data does not contain gold article-frame labels paired to these images, so the reported runs cover image-only and Subtask 2a only. The scores in the report are local held-out scores, not official SemEval scores.

## Run it locally

Use Python 3.11, 3.12 or 3.13. Python 3.12 was used for the reported runs. In a terminal, change into the extracted `Assignment_1_Group_12` folder and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m framing prepare-release --archive data/release1/data.zip --out data/release1
python -m unittest discover -s tests -v
python -m framing audit --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/audit
python -m framing eda --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/eda
python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/classical --model classical --seed 42 --none-mode exclude
python -m pip install -r requirements-pretrained.txt
python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/frozen --model frozen --checkpoint torchvision:resnet18 --seed 42 --none-mode exclude
```

The first command may need `python3.11` or `python3.13` if Python 3.12 is not installed. The frozen run downloads pretrained ResNet-18 weights on first use. Its CPU run can take several minutes.

`prepare-release` checks the image to article joins and extracts the Subtask 2 files needed for training. Inspect `runs/release1/audit/audit.json` for data checks, `runs/release1/eda/eda.pdf` for the regenerated plots and each model folder's `image_only/metrics.json` and `2a_image_text/metrics.json` for scores. The CSVs in `artifacts/` let you inspect the reported results without rerunning the models. Do not use `--enable-2b` with this release.
