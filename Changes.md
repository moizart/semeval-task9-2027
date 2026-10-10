# Change log

## 2026-10-10
- Initialized the project from the Assignment 1 submission plan and the two supplied PDF documents.
- Added a reproducible manifest based workflow, image data audit, selective EDA PDF, grouped internal splits, classical and optional frozen encoder baselines, evaluation and response writing.
- Added a LaTeX report draft and this agent guide.
- Data dependent results remain pending: the dataset ZIP, exact label file, starter kit, and official scorer are not present in the workspace.
- Preserved the source plan as `PLAN.md`. Verified audit, EDA, classical training, prediction, and scoring on a synthetic fixture; ran contract tests and compiled the LaTeX draft. Synthetic fixture scores are not research results.
- Located the official label and trial ZIP URLs in the task PDF, but the configured cloud proxy refused the connection, so neither file could be retrieved.
- Separated the organizer image identifier from its optional local image path so responses contain the required filename when images live in nested folders.
- Added per-frame feature-association output and a text/image context comparison to the audit. Re-ran the EDA workflow and four contract tests successfully.
- Added a guarded macOS setup script for `/Users/abdulmoiz/Documents/Semeval Task9`; it clones or updates the project, creates a virtual environment, installs base dependencies, and runs the contract tests without touching an unrelated nonempty folder.
- Updated the macOS setup script after a Python 3.14 `ensurepip` failure: it now probes working Python 3.11–3.13 installations, prefers 3.12, and rebuilds only a recognized partial virtual environment.
- Verified the revised setup in a clean checkout with Python 3.12, installed current base dependencies, passed all four contract tests, and ran classical training plus response generation on synthetic data.
- Imported the public organizer Subtask 2 trial ZIP (20 image--article pairs, four languages) with exact machine labels from its `labels.txt`; fixed parsing of filename-only unframed gold rows and added validated article/text-frame joins.
- Added a frozen torchvision ResNet-18 encoder option and ran image-only, 2a, and 2b variants for both baseline families on a shared 14/3/3 trial split. Added derived audit, EDA, split, predictions, and metrics snapshots under `artifacts/`; raw news files remain ignored.
- Rewrote the Assignment 1 report with measured trial findings, limitations, verified literature metadata, Group 12 members, and subject-name/code placeholders pending user input. Verified the editor LaTeX compiler and six contract tests.
- Added `PROJECT_GUIDE.md` with the complete Subtask 2 workflow, local Mac setup, outputs, interpretation of trial results, and Assignment 1 submission checklist. The newly attached `data.zip` exceeds the workspace's 32 MiB per-file transfer limit; requested parts at or below 30,000,000 bytes. Full-data inspection, integration, runs, and report updates remain pending receipt of the data parts.
- Re-ran the public-trial import, audit, EDA, classical baseline, frozen ResNet-18 baseline, prediction/TSV scoring smoke check, and six contract tests. All completed after setting `TORCH_HOME` to a writable cache for the sandbox. The six held-out metrics and error counts matched the checked-in trial snapshot; the in-sample prediction smoke score is not an evaluation result. Recompiled `report/main.tex` with the desktop editor successfully.
