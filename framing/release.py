"""Import the attached 400-pair English Subtask 2 training release."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from .data import file_sha256, parse_labels, read_labels


IMAGE_PATTERN = re.compile(r"EN_doc_([1-9][0-9]*)_image\.jpg")
ROOT = "data/subtask2"
LABEL_SOURCE = "data/subtask1/SEMEVAL23_FRAMING_INITIAL_TRAIN-A/labels.txt"
OTHER_LABEL_SOURCE = "data/subtask1/SEMEVAL23_FRAMING_INITIAL_TRAIN-B/labels.txt"


def prepare_release(archive: Path, out: Path) -> dict:
    """Validate the known release layout and extract only Subtask 2 inputs.

    The archive's Subtask 1 articles are a separate corpus, not paired gold
    text-frame labels for these 400 Subtask 2 article/image pairs.
    """
    archive = archive.resolve()
    out = out.resolve()
    extracted = out / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        if len(members) != len(set(members)):
            raise ValueError("Duplicate ZIP entry names")
        required = {f"{ROOT}/subtask-2-annotations.txt", LABEL_SOURCE, OTHER_LABEL_SOURCE}
        missing = required - set(members)
        if missing:
            raise ValueError(f"Missing release metadata: {sorted(missing)}")
        labels_a = [line.strip() for line in bundle.read(LABEL_SOURCE).decode("utf-8-sig").splitlines() if line.strip()]
        labels_b = [line.strip() for line in bundle.read(OTHER_LABEL_SOURCE).decode("utf-8-sig").splitlines() if line.strip()]
        if labels_a != labels_b or len(labels_a) != 14 or len(labels_a) != len(set(labels_a)):
            raise ValueError("The release's 14-label vocabularies disagree or are invalid")
        annotation = bundle.read(f"{ROOT}/subtask-2-annotations.txt").decode("utf-8-sig").splitlines()
        if len(annotation) != 400:
            raise ValueError(f"Expected 400 image annotations, found {len(annotation)}")
        available = set(members)
        rows = []
        needed_files = set()
        document_ids = set()
        for line_number, line in enumerate(annotation, 1):
            fields = line.split("\t")
            image_name = fields[0]
            match = IMAGE_PATTERN.fullmatch(image_name)
            if match is None:
                raise ValueError(f"Invalid image identifier on annotation line {line_number}: {image_name!r}")
            document_id = int(match.group(1))
            if document_id in document_ids:
                raise ValueError(f"Duplicate document ID {document_id}")
            document_ids.add(document_id)
            image_labels = [value for value in fields[1:] if value]
            parse_labels("|".join(image_labels), labels_a)
            article_name = f"EN_M_TRAIN_doc_{document_id}.txt"
            image_member = f"{ROOT}/images/{image_name}"
            article_member = f"{ROOT}/images/{article_name}"
            for member in (image_member, article_member):
                if member not in available:
                    raise FileNotFoundError(f"Missing paired file for {image_name}: {member}")
                needed_files.add(member)
            rows.append({
                "image": image_name,
                "image_path": str((Path("extracted") / image_member).as_posix()),
                "article": str((Path("extracted") / article_member).as_posix()),
                "language": "EN",
                "split": "train",
                "image_labels": "|".join(image_labels),
                "group": f"EN_doc_{document_id}",
                "source": "",
                "dataset": "attached_subtask2_english_chunk1",
            })
        if document_ids != set(range(1, 401)):
            raise ValueError("Annotation document IDs must be exactly 1 through 400")
        expected_images = {name for name in available if name.startswith(f"{ROOT}/images/") and name.lower().endswith(".jpg")}
        expected_articles = {name for name in available if name.startswith(f"{ROOT}/images/") and name.lower().endswith(".txt")}
        if needed_files != expected_images | expected_articles or len(expected_images) != 400 or len(expected_articles) != 400:
            raise ValueError("Subtask 2 image/article files do not match the annotations exactly")

        for member in sorted(needed_files):
            destination = extracted / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        out.mkdir(parents=True, exist_ok=True)
        (out / "framing-labels.txt").write_text("\n".join(labels_a) + "\n", encoding="utf-8")
        read_labels(out / "framing-labels.txt")
        pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)

    provenance = {
        "archive": archive.name,
        "archive_sha256": file_sha256(archive),
        "scope": "Attached Subtask 2 English training release, chunk 1; 400 labeled article-image pairs",
        "examples": len(rows),
        "labels": len(labels_a),
        "label_source": LABEL_SOURCE,
        "article_join": "Verified shared numeric document ID: EN_doc_N_image.jpg -> EN_M_TRAIN_doc_N.txt",
        "text_frame_labels": "Unavailable for these Subtask 2 articles; separate Subtask 1 corpus is not joined",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
