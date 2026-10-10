from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


REQUIRED = {"image", "article", "language", "split", "image_labels"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Label file is empty or contains duplicates")
    return labels


def parse_labels(value: str, vocabulary: list[str]) -> list[str]:
    result = [part.strip() for part in str(value).split("|") if part.strip()]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate label in {value!r}")
    unknown = set(result) - set(vocabulary)
    if unknown:
        raise ValueError(f"Unknown label(s): {sorted(unknown)}")
    return result


def read_manifest(path: Path, vocabulary: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty or frame.image.duplicated().any():
        raise ValueError("Manifest is empty or image names are duplicated")
    if not set(frame.split).issubset({"train", "dev", "test"}):
        raise ValueError("Split must be train, dev, or test")
    if not set(frame.language).issubset({"BG", "EN", "PT", "RU"}):
        raise ValueError("Language must be BG, EN, PT, or RU")
    for col in ("text_labels", "group", "source"):
        if col not in frame:
            frame[col] = ""
    if "image_path" not in frame:
        frame["image_path"] = frame["image"]
    frame["image_path"] = frame["image_path"].where(frame["image_path"] != "", frame["image"])
    for column in ("image_labels", "text_labels"):
        for value in frame[column]:
            parse_labels(value, vocabulary)
    frame["group"] = frame["group"].where(frame["group"] != "", frame["article"])
    if (frame["group"] == "").any():
        raise ValueError("Each row needs an article or group")
    return frame


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    full = path if path.is_absolute() else root / path
    full = full.resolve()
    if not full.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data directory: {value}")
    return full


def article_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.seek(0)
        return ImageOps.exif_transpose(source).convert("RGB")


def multilabel(frame: pd.DataFrame, column: str, vocabulary: list[str]) -> np.ndarray:
    label_columns = {label: column_number for column_number, label in enumerate(vocabulary)}
    targets = np.zeros((len(frame), len(vocabulary)), dtype=np.uint8)
    for row_number, raw_labels in enumerate(frame[column]):
        for label in parse_labels(raw_labels, vocabulary):
            targets[row_number, label_columns[label]] = 1
    return targets


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
