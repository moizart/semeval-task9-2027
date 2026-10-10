# Organizer trial outputs

These files are reproducible **trial-data checks**, not official competition results. The public trial archive has 20 images; the held-out evaluation has only three images. Raw news text and images are not redistributed here. Download the organizer ZIP and run the commands in the repository README to reproduce them.

- `trial-provenance.json`: archive URL, SHA-256, and import assumptions.
- `trial-audit.json`: file validation, language counts, label counts, and text/image comparison.
- `trial-eda.pdf`: the required single EDA plots PDF for the trial.
- `trial-frame-feature-associations.csv`: exploratory standardized feature differences.
- `trial-split-manifest.csv`: exact fit, calibration, and evaluation assignments.
- `trial-metrics.csv`: the six classical/frozen input variants on the same held-out split.
- `trial-*-versions.json`: Python and package versions used for each baseline family.
- `trial-*-image-only-predictions.csv`: representative evaluation predictions and probabilities.

The official scorer and full training data were not available, so these outputs must be replaced before submitting Assignment 1 as a full-data study.
