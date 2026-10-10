The user-supplied English Subtask 2 training release is committed at `release1/data.zip`, with its manifest and provenance; see `release1/README.md`. Working extracts and experiment runs remain ignored by Git.

For the public trial, download the organizer ZIP as `data/trial.zip`, then run `python -m framing prepare-trial --archive data/trial.zip --out data` from the repository root. The importer creates `data/manifest.csv` and an exact machine label list at `data/framing-labels.txt`. See the main README for the full commands.
