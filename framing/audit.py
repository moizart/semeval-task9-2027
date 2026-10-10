from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image

from .data import article_text, file_sha256, image_rgb, multilabel, resolve
from .features import FEATURE_NAMES, classical_features


def collect(frame: pd.DataFrame, root: Path, labels: list[str]) -> tuple[pd.DataFrame, dict]:
    feature_records = []
    image_hashes = defaultdict(list)
    article_hashes = defaultdict(list)
    errors = []
    for example in frame.itertuples():
        record = {"image": example.image, "split": example.split, "language": example.language,
                  "source": example.source, "group": example.group,
                  "label_count": len(example.image_labels.split("|")) if example.image_labels else 0}
        try:
            image_path = resolve(root, example.image_path)
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                raise ValueError("Unsupported image extension")
            with Image.open(image_path) as image_file:
                image_file.verify()
            record.update(zip(FEATURE_NAMES, classical_features(image_path)))
            image_hashes[file_sha256(image_path)].append(example.image)
            small_image = np.asarray(image_rgb(image_path).resize((9, 8)).convert("L"))
            record["dhash"] = "".join(map(str, (small_image[:, 1:] > small_image[:, :-1]).astype(int).flat))
        except Exception as error:
            errors.append({"image": example.image, "file": example.image, "error": str(error)})
        try:
            article_path = resolve(root, example.article)
            text = article_text(article_path)
            record["article_chars"] = len(text)
            record["article_words"] = len(text.split())
            article_hashes[file_sha256(article_path)].append(example.article)
        except Exception as error:
            errors.append({"image": example.image, "file": example.article, "error": str(error)})
        feature_records.append(record)
    statistics = pd.DataFrame(feature_records)
    image_targets = multilabel(frame, "image_labels", labels)
    training_rows = frame.split == "train"
    duplicates = {"images": [names for names in image_hashes.values() if len(names) > 1],
                  "articles": [sorted(set(names)) for names in article_hashes.values() if len(set(names)) > 1]}
    # matching hashes flag images that still need a human check.
    same_image_hashes = statistics.dropna(subset=["dhash"]).groupby("dhash").image.apply(list)
    report = {
        "rows_by_split_language": frame.groupby(["split", "language"]).size().rename("count").reset_index().to_dict("records"),
        "train_label_counts": dict(zip(labels, image_targets[training_rows].sum(axis=0).astype(int).tolist())),
        "train_label_cardinality": Counter(image_targets[training_rows].sum(axis=1).astype(int).tolist()),
        "errors": errors,
        "exact_duplicates": duplicates,
        "same_dhash_candidates": [names for names in same_image_hashes if len(names) > 1],
        "train_dev_group_overlap": sorted(set(frame.loc[training_rows, "group"]) & set(frame.loc[~training_rows, "group"])),
    }
    if (frame.loc[training_rows, "text_labels"] != "").any():
        text_targets = multilabel(frame.loc[training_rows], "text_labels", labels)
        training_image_targets = image_targets[training_rows]
        report["text_image_context_comparison"] = {
            "rows_with_any_text_frame": int((text_targets.sum(axis=1) > 0).sum()),
            "rows_with_any_image_frame": int((training_image_targets.sum(axis=1) > 0).sum()),
            "rows_with_any_shared_frame": int(np.logical_and(text_targets, training_image_targets).any(axis=1).sum()),
            "rows_with_different_frame_sets": int((text_targets != training_image_targets).any(axis=1).sum()),
        }
    report["train_label_cardinality"] = {
        str(label_count): count for label_count, count in report["train_label_cardinality"].items()
    }
    return statistics, report


def save_audit(frame: pd.DataFrame, root: Path, labels: list[str], out: Path) -> tuple[pd.DataFrame, dict]:
    out.mkdir(parents=True, exist_ok=True)
    stats, report = collect(frame, root, labels)
    stats.to_csv(out / "features.csv", index=False)
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return stats, report


def save_eda(frame: pd.DataFrame, stats: pd.DataFrame, labels: list[str], out: Path) -> None:
    training_rows = frame.split == "train"
    training_stats = stats.loc[training_rows].reset_index(drop=True)
    image_targets = multilabel(frame.loc[training_rows], "image_labels", labels)
    if training_stats.empty:
        raise ValueError("No training rows for EDA")
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "eda.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].barh(labels, image_targets.sum(axis=0)); axes[0].set_title("Image frame frequency (training)"); axes[0].set_xlabel("Images")
        label_counts = np.bincount(image_targets.sum(axis=1).astype(int), minlength=len(labels) + 1)
        axes[1].bar(np.arange(len(label_counts)), label_counts); axes[1].set_title("Image labels per example (training)"); axes[1].set_xlabel("Number of labels"); axes[1].set_ylabel("Images")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 7))
        cooccurrence = image_targets.astype(np.int64).T @ image_targets.astype(np.int64)
        heat = ax.imshow(cooccurrence, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90); ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Image frame co-occurrence (training image counts)")
        fig.colorbar(heat, ax=ax, label="Images"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        feature_groups = [
            ("log_aspect", "Image aspect ratio (log width/height)", "log width / height"),
            ("brightness_mean", "Mean image brightness", "Mean grayscale intensity (0–1)"),
            ("saturation_mean", "Mean image saturation", "Mean saturation (0–1)"),
            ("edge_density", "Image edge density", "Fraction of pixels with edge response > 0.2"),
            ("article_words", "Article length", "Whitespace-separated words"),
        ]
        for column, title, x_label in feature_groups:
            if column not in training_stats or training_stats[column].dropna().empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(training_stats[column].dropna(), bins=30)
            ax.set(title=f"{title} (training)", xlabel=x_label, ylabel="Images")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # these differences suggest checks, not proven framing cues.
        feature_names = [name for name, _, _ in feature_groups]
        associations = np.full((len(labels), len(feature_names)), np.nan)
        for feature_index, column in enumerate(feature_names):
            values = training_stats[column].to_numpy(dtype=float)
            scale = np.nanstd(values)
            if scale == 0 or not np.isfinite(scale):
                continue
            for label_index in range(len(labels)):
                positive = values[image_targets[:, label_index] == 1]
                negative = values[image_targets[:, label_index] == 0]
                if len(positive) and len(negative):
                    associations[label_index, feature_index] = (np.nanmean(positive) - np.nanmean(negative)) / scale
        pd.DataFrame(associations, index=labels, columns=feature_names).to_csv(out / "frame_feature_associations.csv")
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
        heat = ax.imshow(associations, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Frame associations: positive minus negative mean / feature SD")
        fig.colorbar(heat, ax=ax, label="Standardized mean difference")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        training_stats.language.value_counts().sort_index().plot.bar(ax=ax)
        ax.set(title="Training images by language", xlabel="Language", ylabel="Images")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        if (training_stats.source != "").any():
            fig, ax = plt.subplots(figsize=(10, 5))
            training_stats.source.value_counts().head(20).plot.bar(ax=ax)
            ax.set(title="Top 20 training sources", xlabel="Source", ylabel="Images")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
