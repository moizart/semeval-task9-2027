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
    records = []
    image_hashes = defaultdict(list)
    article_hashes = defaultdict(list)
    errors = []
    for row in frame.itertuples():
        record = {"image": row.image, "split": row.split, "language": row.language,
                  "source": row.source, "group": row.group, "label_count": len(row.image_labels.split("|")) if row.image_labels else 0}
        try:
            path = resolve(root, row.image_path)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                raise ValueError("Unsupported image extension")
            with Image.open(path) as img:
                img.verify()
            record.update(zip(FEATURE_NAMES, classical_features(path)))
            image_hashes[file_sha256(path)].append(row.image)
            tiny = np.asarray(image_rgb(path).resize((9, 8)).convert("L"))
            record["dhash"] = "".join(map(str, (tiny[:, 1:] > tiny[:, :-1]).astype(int).flat))
        except Exception as exc:
            errors.append({"image": row.image, "file": row.image, "error": str(exc)})
        try:
            path = resolve(root, row.article)
            text = article_text(path)
            record["article_chars"] = len(text)
            record["article_words"] = len(text.split())
            article_hashes[file_sha256(path)].append(row.article)
        except Exception as exc:
            errors.append({"image": row.image, "file": row.article, "error": str(exc)})
        records.append(record)
    stats = pd.DataFrame(records)
    y = multilabel(frame, "image_labels", labels)
    train = frame.split == "train"
    duplicates = {"images": [v for v in image_hashes.values() if len(v) > 1],
                  "articles": [sorted(set(v)) for v in article_hashes.values() if len(set(v)) > 1]}
    # A dHash match is only a candidate; templates and resizes require manual inspection.
    near = stats.dropna(subset=["dhash"]).groupby("dhash").image.apply(list)
    report = {
        "rows_by_split_language": frame.groupby(["split", "language"]).size().rename("count").reset_index().to_dict("records"),
        "train_label_counts": dict(zip(labels, y[train].sum(axis=0).astype(int).tolist())),
        "train_label_cardinality": Counter(y[train].sum(axis=1).astype(int).tolist()),
        "errors": errors,
        "exact_duplicates": duplicates,
        "same_dhash_candidates": [v for v in near if len(v) > 1],
        "train_dev_group_overlap": sorted(set(frame.loc[train, "group"]) & set(frame.loc[~train, "group"])),
    }
    if (frame.loc[train, "text_labels"] != "").any():
        text_y = multilabel(frame.loc[train], "text_labels", labels)
        image_y = y[train]
        report["text_image_context_comparison"] = {
            "rows_with_any_text_frame": int((text_y.sum(axis=1) > 0).sum()),
            "rows_with_any_image_frame": int((image_y.sum(axis=1) > 0).sum()),
            "rows_with_any_shared_frame": int(np.logical_and(text_y, image_y).any(axis=1).sum()),
            "rows_with_different_frame_sets": int((text_y != image_y).any(axis=1).sum()),
        }
    report["train_label_cardinality"] = {str(k): v for k, v in report["train_label_cardinality"].items()}
    return stats, report


def save_audit(frame: pd.DataFrame, root: Path, labels: list[str], out: Path) -> tuple[pd.DataFrame, dict]:
    out.mkdir(parents=True, exist_ok=True)
    stats, report = collect(frame, root, labels)
    stats.to_csv(out / "features.csv", index=False)
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return stats, report


def save_eda(frame: pd.DataFrame, stats: pd.DataFrame, labels: list[str], out: Path) -> None:
    train = frame.split == "train"
    sample = stats.loc[train].reset_index(drop=True)
    y = multilabel(frame.loc[train], "image_labels", labels)
    if sample.empty:
        raise ValueError("No training rows for EDA")
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "eda.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].barh(labels, y.sum(axis=0)); axes[0].set_title("Image frame frequency (training)"); axes[0].set_xlabel("Images")
        counts = np.bincount(y.sum(axis=1).astype(int), minlength=len(labels) + 1)
        axes[1].bar(np.arange(len(counts)), counts); axes[1].set_title("Image labels per example (training)"); axes[1].set_xlabel("Number of labels"); axes[1].set_ylabel("Images")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 7))
        both = y.astype(np.int64).T @ y.astype(np.int64)
        heat = ax.imshow(both, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90); ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Image frame co-occurrence (training image counts)")
        fig.colorbar(heat, ax=ax, label="Images"); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        groups = [
            ("log_aspect", "Image aspect ratio (log width/height)", "log width / height"),
            ("brightness_mean", "Mean image brightness", "Mean grayscale intensity (0–1)"),
            ("saturation_mean", "Mean image saturation", "Mean saturation (0–1)"),
            ("edge_density", "Image edge density", "Fraction of pixels with edge response > 0.2"),
            ("article_words", "Article length", "Whitespace-separated words"),
        ]
        for column, title, xlab in groups:
            if column not in sample or sample[column].dropna().empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(sample[column].dropna(), bins=30)
            ax.set(title=f"{title} (training)", xlabel=xlab, ylabel="Images")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # Standardized positive-minus-negative means reveal simple associations,
        # but can also reflect language, source, or repeated-template confounds.
        columns = [name for name, _, _ in groups]
        effect = np.full((len(labels), len(columns)), np.nan)
        for j, column in enumerate(columns):
            values = sample[column].to_numpy(dtype=float)
            scale = np.nanstd(values)
            if scale == 0 or not np.isfinite(scale):
                continue
            for i in range(len(labels)):
                positive = values[y[:, i] == 1]
                negative = values[y[:, i] == 0]
                if len(positive) and len(negative):
                    effect[i, j] = (np.nanmean(positive) - np.nanmean(negative)) / scale
        pd.DataFrame(effect, index=labels, columns=columns).to_csv(out / "frame_feature_associations.csv")
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
        heat = ax.imshow(effect, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title("Frame associations: positive minus negative mean / feature SD")
        fig.colorbar(heat, ax=ax, label="Standardized mean difference")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        sample.language.value_counts().sort_index().plot.bar(ax=ax)
        ax.set(title="Training images by language", xlabel="Language", ylabel="Images")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        if (sample.source != "").any():
            fig, ax = plt.subplots(figsize=(10, 5))
            sample.source.value_counts().head(20).plot.bar(ax=ax)
            ax.set(title="Top 20 training sources", xlabel="Source", ylabel="Images")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
