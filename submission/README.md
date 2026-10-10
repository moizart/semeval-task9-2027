# Assignment 1 submission | Group 12

**Subject:** Reinforcement Learning (CS-317)  
**Project:** SemEval 2027 Task 9, Subtask 2, image framing in news  
**Overleaf project link:** **[PASTE YOUR OVERLEAF LINK HERE BEFORE SUBMITTING]**

**Members:** Abdul Moiz (CMS 521568), Muhammad Ibsham Tariq (CMS 502451), Abdullah Khalil (CMS 501492) and Zia Ur Rehman (CMS 511386).

## Start here

| File or folder | What to look for |
| --- | --- |
| `report/Assignment_1_Report_Group_12.pdf` | The compiled Assignment 1 report with its References section. |
| `report/main.tex`, `report/references.bib` | Editable LaTeX and BibTeX sources for Overleaf. |
| `artifacts/release1-eda.pdf` | The one PDF of selected EDA plots requested in the guidelines. |
| `artifacts/release1-metrics.csv` | Four held-out baseline results on the attached English training chunk. |
| `artifacts/release1-audit.json`, `artifacts/release1-split-manifest.csv` | Data checks and the saved fit/calibration/evaluation split. |
| `data/release1/data.zip` | The original 400-pair English Subtask 2 training release supplied for this assignment. |
| `data/release1/manifest.csv`, `framing-labels.txt`, `provenance.json` | Image/article mapping, exact frame names and archive hash. |
| `framing/` | Import, audit, feature extraction, training, prediction and scoring code. |
| `tests/` | Contract checks for data joins, splits and response parsing. |
| `PROJECT_GUIDE.md` | Full explanation of the task, implementation, results and assignment requirements. |
| `Agent.md`, `Changes.md`, `PLAN.md` | Working guide, dated change history and original project plan. |

The source ZIP includes the original data archive. Generated model files and working extracts are left out because the commands below reproduce them. The report's results are from **training chunk 1**, which has 400 English image/article pairs. They are local held-out results, not official SemEval scores. This chunk has no gold article-frame labels paired to the Subtask 2 images, so it cannot support a valid 2b experiment. The separate Subtask 1 corpus in the data ZIP is not used as a substitute.

## Run the code after extracting this submission ZIP

Use Python 3.11, 3.12 or 3.13. Python 3.12 was used for the saved runs. From the extracted `Assignment_1_Group_12` folder:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m framing prepare-release --archive data/release1/data.zip --out data/release1
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python -m framing audit --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/audit
./.venv/bin/python -m framing eda --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/eda
./.venv/bin/python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/classical --model classical --seed 42 --none-mode exclude
./.venv/bin/python -m pip install -r requirements-pretrained.txt
./.venv/bin/python -m framing train --manifest data/release1/manifest.csv --labels data/release1/framing-labels.txt --out runs/release1/frozen --model frozen --checkpoint torchvision:resnet18 --seed 42 --none-mode exclude
```

The frozen run downloads pretrained ResNet-18 weights the first time it runs. Open `runs/release1/classical/2a_image_text/metrics.json` and `runs/release1/frozen/2a_image_text/metrics.json` for the measured scores. Each run also saves evaluation predictions and model files. The tracked `artifacts/` folder has the corresponding results without requiring a rerun. Do not use `--enable-2b` for this release.

For a Git checkout on Abdul Moiz's Mac, `scripts/setup-mac.sh` can handle the environment after the repository is cloned. **Use the manual commands above for this extracted submission ZIP**, since it has no Git metadata.

## Before uploading to the course portal

1. Replace the Overleaf placeholder at the top of this file with your real project link.
2. Upload `report/main.tex` and `report/references.bib` to Overleaf if the instructor wants access to the editable report. The PDF already includes the bibliography.
3. Confirm the required filename, portal and due date with the instructor. The supplied Assignment 1 guideline PDF does not specify them.

The five related papers discussed in the report are Card et al. (2015), Akyürek et al. (2020), Tourni et al. (2021), Piskorski et al. (2023) and Arora et al. (2025). He et al. (2016) supports the ResNet baseline. The report also cites the organizer's task description. Full bibliographic details are in `report/references.bib` and the report PDF.
