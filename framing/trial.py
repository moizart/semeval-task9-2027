"""Import the organizer's public Subtask 2 trial archive without guessing joins."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .data import file_sha256, parse_labels, read_labels


LANGUAGES = ("BG", "EN", "PT", "RU")


def _extract_checked(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe path in archive: {entry.filename}")
        bundle.extractall(destination)


def _text_frames(path: Path, vocabulary: list[str]) -> dict[str, set[str]]:
    by_article: dict[str, set[str]] = defaultdict(set)
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) < 4:
            raise ValueError(f"Malformed text annotation {path}:{line_number}")
        by_article[fields[0]].update(label for label in fields[3:] if label)
    for article, values in by_article.items():
        parse_labels("|".join(sorted(values)), vocabulary)
    return by_article


def prepare_trial(archive: Path, out: Path) -> dict:
    archive = archive.resolve()
    out = out.resolve()
    extracted = out / "trial" / "extracted"
    _extract_checked(archive, extracted)
    labels_source = extracted / "labels.txt"
    vocabulary = read_labels(labels_source)
    if len(vocabulary) != 14:
        raise ValueError(f"Expected 14 organizer labels, found {len(vocabulary)}")
    rows = []
    for language in LANGUAGES:
        folder = extracted / language
        text_labels = _text_frames(folder / "subtask-1-annotations.txt", vocabulary)
        gold_path = folder / "subtask-2-annotations.txt"
        for line_number, line in enumerate(gold_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.split("\t")
            # Trial gold omits the trailing tab for an unframed image.
            if not fields[0]:
                raise ValueError(f"Malformed image annotation {gold_path}:{line_number}")
            image_name = fields[0]
            image_labels = [value for value in fields[1:] if value]
            parse_labels("|".join(image_labels), vocabulary)
            article_name = Path(image_name).stem + ".txt"
            image_path = folder / "raw-documents" / image_name
            article_path = folder / "raw-documents" / article_name
            if not image_path.is_file() or not article_path.is_file():
                raise FileNotFoundError(f"Image/article pair missing for {image_name}")
            if article_name not in text_labels:
                raise ValueError(f"Text annotations missing for {article_name}")
            rows.append({
                "image": image_name,
                "image_path": str(image_path.relative_to(out)),
                "article": str(article_path.relative_to(out)),
                "language": language,
                "split": "train",
                "image_labels": "|".join(image_labels),
                "text_labels": "|".join(label for label in vocabulary if label in text_labels[article_name]),
                "group": f"{language}:{Path(image_name).stem}",
                "source": "",
                "dataset": "organizer_trial_2026-09-04",
            })
    frame = pd.DataFrame(rows)
    if frame.image.duplicated().any():
        raise ValueError("Duplicate image names in organizer trial archive")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(labels_source, out / "framing-labels.txt")
    frame.to_csv(out / "manifest.csv", index=False)
    provenance = {
        "archive": archive.name,
        "archive_sha256": file_sha256(archive),
        "source": "https://propaganda.math.unipd.it/semeval2027task9/data/trialdataSemEval2027Task9.zip",
        "examples": len(frame),
        "labels": len(vocabulary),
        "scope": "Public organizer trial data; not the full competition training set",
        "text_labels": "Union of paragraph frames from subtask-1-annotations.txt, used only as Subtask 2b context",
    }
    (out / "trial-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
