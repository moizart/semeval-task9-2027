from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import metric_bundle


def read_tsv(path: Path, labels: list[str]) -> dict[str, set[str]]:
    image_labels = {}
    allowed = set(labels)
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        fields = line.split("\t")
        image = fields[0]
        # trial gold sometimes leaves the empty field out.
        frame_labels = {item for item in fields[1:] if item}
        if not image or image in image_labels:
            raise ValueError(f"Empty or duplicate image name at line {line_number}")
        if len(frame_labels) != len([item for item in fields[1:] if item]):
            raise ValueError(f"Duplicate label at line {line_number}")
        if not frame_labels.issubset(allowed):
            raise ValueError(f"Unknown labels at line {line_number}: {frame_labels - allowed}")
        image_labels[image] = frame_labels
    return image_labels


def score_tsv(gold: Path, response: Path, labels: list[str], none_mode: str) -> dict:
    reference = read_tsv(gold, labels)
    predicted = read_tsv(response, labels)
    if set(reference) != set(predicted):
        raise ValueError(f"Image names differ: {len(set(reference)-set(predicted))} missing, {len(set(predicted)-set(reference))} extra")
    image_names = list(reference)
    targets = np.asarray([[label in reference[name] for label in labels] for name in image_names], dtype=np.uint8)
    predictions = np.asarray([[label in predicted[name] for label in labels] for name in image_names], dtype=np.uint8)
    return metric_bundle(targets, predictions, labels, none_mode)
