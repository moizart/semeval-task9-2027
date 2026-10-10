# Project agent guide

## Goal
Build a reproducible Assignment 1 package for SemEval 2027 Task 9, Subtask 2 (image framing), covering data audit, selective EDA, a classical image baseline, a frozen vision encoder baseline, evaluation, and a LaTeX report. Keep Subtask 2a and 2b distinct.

## Rules for work on this repository
- Read `Changes.md` before changing the project and append a dated entry for material changes, checks, and known blockers.
- Treat the supplied dataset as read-only. Put raw working data under `data/`, which is ignored by Git by default, and generated artifacts under `runs/`. If the user explicitly asks to commit a dataset, inspect it and track only the intended files.
- Use exact strings from the gold annotation vocabulary. In the public trial ZIP this is `labels.txt` with underscores; the task page's `framing-labels.txt` uses different readable spelling. Verify the vocabulary again for the full training data.
- Do not use test or organizer development scores to select features, thresholds, or hyperparameters. Make grouped internal train, calibration, and evaluation splits from labeled training data.
- Never equate article text frame labels with image frame labels. The former are optional context features for 2b only.
- State clearly when a result is pending because data, the official scorer, or a pretrained checkpoint is unavailable. Do not fabricate counts or scores.
- Prefer a small, understandable baseline over untracked experiments. Save the manifest, split assignments, configuration, package versions, and predictions for each run.

## Current workflow
See `README.md` for commands, `PLAN.md` for the original implementation plan, and `report/main.tex` for the report draft.
