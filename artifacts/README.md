# Reproducible Subtask 2 outputs

## Attached English training release, chunk 1

The `release1-*` files are derived from the user-supplied 400-pair archive in `data/release1/data.zip`. They describe **one fixed grouped internal split**, not official competition results. `release1-provenance.json` records its SHA-256 and join rule; `release1-audit.json` and `release1-eda.pdf` contain data checks and the required single EDA plots PDF; `release1-frame-feature-associations.csv` contains descriptive standardized mean differences. `release1-split-manifest.csv` fixes the 280/60/60 fit/calibration/evaluation assignment. `release1-metrics.csv` compares four image-only/2a baseline configurations. `release1-*-versions.json` and `release1-*-2a-predictions.csv` record environments and representative held-out probabilities/errors.

The archive contains no paired gold article-frame labels for these 400 Subtask 2 examples, so no full-release 2b metric is present. The organizer scorer is still unavailable; these are local scores. Run `python -m framing prepare-release --archive data/release1/data.zip --out data/release1` before reproducing the runs.

## Public organizer trial

These files are reproducible **trial-data checks**, not official competition results. The public trial archive has 20 images; the held-out evaluation has only three images. Raw news text and images are not redistributed here. Download the organizer ZIP and run the commands in the repository README to reproduce them.

- `trial-provenance.json`: archive URL, SHA-256, and import assumptions.
- `trial-audit.json`: file validation, language counts, label counts, and text/image comparison.
- `trial-eda.pdf`: the required single EDA plots PDF for the trial.
- `trial-frame-feature-associations.csv`: exploratory standardized feature differences.
- `trial-split-manifest.csv`: exact fit, calibration, and evaluation assignments.
- `trial-metrics.csv`: the six classical/frozen input variants on the same held-out split.
- `trial-*-versions.json`: Python and package versions used for each baseline family.
- `trial-*-image-only-predictions.csv`: representative evaluation predictions and probabilities.

The trial's three-image evaluation is superseded for Assignment 1 analysis by the attached 400-pair English release above. Neither set has an organizer development/test score.
