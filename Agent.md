# Project agent guide

## Goal
Build a reproducible Assignment 1 package for SemEval 2027 Task 9, Subtask 2 (image framing), covering data audit, selective EDA, a classical image baseline, a frozen vision encoder baseline, evaluation, and a LaTeX report. Keep Subtask 2a and 2b distinct.

## Rules for work on this repository
- Read `Changes.md` before changing the project and append a dated entry for material changes, checks, and known blockers.
- Treat the supplied dataset as read-only. Put raw working data under `data/`, which is ignored by Git by default, and generated artifacts under `runs/`. If the user explicitly asks to commit a dataset, inspect it and track only the intended files.
- Use exact strings from the gold annotation vocabulary. The public trial ZIP's `labels.txt` and the attached release's two Subtask 1 `labels.txt` files agree on 14 underscore-separated machine names; the task page's readable `framing-labels.txt` does not match the gold strings. Verify vocabulary again for later releases.
- Do not use test or organizer development scores to select features, thresholds, or hyperparameters. Make grouped internal train, calibration, and evaluation splits from labeled training data.
- Never equate article text frame labels with image frame labels. The former are optional context features for 2b only. The attached English Subtask 2 chunk has no paired gold text-frame labels; its separate Subtask 1 corpus cannot be joined to those 400 images.
- State clearly when a result is pending because later data, paired 2b text frames, or the official scorer are unavailable. Do not fabricate counts or scores.
- Prefer a small, understandable baseline over untracked experiments. Save the manifest, split assignments, configuration, package versions, and predictions for each run.

## Current workflow
See `README.md` and `PROJECT_GUIDE.md` for commands, `PLAN.md` for the original implementation plan and `report/main.tex` for the report source. `submission/README.md` is the upload package guide and includes the supplied Overleaf link. The ZIP uses the user-exported `report/A1_Semeval_Task9_Group12_Report.pdf`. After changing any submission input, run `python scripts/build_submission.py` and check the fresh ZIP under `dist/`.
