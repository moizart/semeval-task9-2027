from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import metric_bundle


def read_tsv(path: Path, labels: list[str]) -> dict[str, set[str]]:
    rows = {}
    allowed = set(labels)
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        fields = raw.split("\t")
        image = fields[0]
        if len(fields) < 2:
            raise ValueError(f"Missing tab-separated framing field at line {number}")
        values = {item for item in fields[1:] if item}
        if not image or image in rows:
            raise ValueError(f"Empty or duplicate image name at line {number}")
        if len(values) != len([item for item in fields[1:] if item]):
            raise ValueError(f"Duplicate label at line {number}")
        if not values.issubset(allowed):
            raise ValueError(f"Unknown labels at line {number}: {values - allowed}")
        rows[image] = values
    return rows


def score_tsv(gold: Path, response: Path, labels: list[str], none_mode: str) -> dict:
    reference = read_tsv(gold, labels)
    predicted = read_tsv(response, labels)
    if set(reference) != set(predicted):
        raise ValueError(f"Image names differ: {len(set(reference)-set(predicted))} missing, {len(set(predicted)-set(reference))} extra")
    names = list(reference)
    y = np.asarray([[label in reference[name] for label in labels] for name in names], dtype=np.uint8)
    p = np.asarray([[label in predicted[name] for label in labels] for name in names], dtype=np.uint8)
    return metric_bundle(y, p, labels, none_mode)
