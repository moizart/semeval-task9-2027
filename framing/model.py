from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data import article_text, multilabel, resolve
from .features import classical_features


def split_groups(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """Search grouped 70/15/15 splits for reasonable label/language balance."""
    if frame.group.nunique() < 3:
        raise ValueError("At least three independent groups are needed for fit, calibration, evaluation")
    best = None
    target = y.mean(axis=0)
    languages = pd.get_dummies(frame.language).to_numpy(dtype=float)
    lang_target = languages.mean(axis=0)
    for trial in range(100):
        try:
            first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + trial)
            fit, remaining = next(first.split(frame, groups=frame.group))
            second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1000 + trial)
            cal_local, eval_local = next(second.split(frame.iloc[remaining], groups=frame.group.iloc[remaining]))
        except ValueError:
            continue
        parts = [fit, remaining[cal_local], remaining[eval_local]]
        if min(map(len, parts)) == 0:
            continue
        score = sum(np.abs(y[idx].mean(axis=0) - target).mean() +
                    np.abs(languages[idx].mean(axis=0) - lang_target).mean() for idx in parts)
        if best is None or score < best[0]:
            best = score, parts
    if best is None:
        raise ValueError("Could not create three nonempty grouped partitions")
    names = np.empty(len(frame), dtype=object)
    for idx, name in zip(best[1], ["fit", "calibration", "evaluation"]):
        names[idx] = name
    return names


def frozen_features(paths: list[Path], checkpoint: str, batch_size: int = 16) -> np.ndarray:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("Frozen baseline needs requirements-pretrained.txt") from exc
    from .data import image_rgb
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = [image_rgb(p) for p in paths[start:start + batch_size]]
            inputs = processor(images=images, return_tensors="pt")
            if hasattr(model, "get_image_features"):
                result = model.get_image_features(**inputs)
                if hasattr(result, "pooler_output"):
                    result = result.pooler_output
            else:
                output = model(**inputs)
                result = output.pooler_output if output.pooler_output is not None else output.last_hidden_state.mean(dim=1)
            chunks.append(result.float().cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def image_features(frame: pd.DataFrame, root: Path, kind: str, checkpoint: str | None) -> np.ndarray:
    paths = [resolve(root, value) for value in frame.image_path]
    if kind == "classical":
        return np.vstack([classical_features(p) for p in paths])
    if not checkpoint:
        raise ValueError("--checkpoint is required for frozen model")
    return frozen_features(paths, checkpoint)


def get_text(frame: pd.DataFrame, root: Path) -> list[str]:
    return [article_text(resolve(root, value)) for value in frame.article]


def build_matrix(frame: pd.DataFrame, root: Path, image: np.ndarray, labels: list[str], mode: str,
                 fit_indices: np.ndarray | None = None, preprocess: dict | None = None) -> tuple[sparse.csr_matrix, dict]:
    if preprocess is None:
        scaler = StandardScaler().fit(image[fit_indices])
        preprocess = {"scaler": scaler, "vectorizer": None}
        if mode != "image_only":
            # Fit vocabulary only on fit examples to prevent information leakage.
            vectorizer = TfidfVectorizer(max_features=20000, min_df=1, ngram_range=(1, 2), sublinear_tf=True)
            vectorizer.fit([get_text(frame.iloc[[i]], root)[0] for i in fit_indices])
            preprocess["vectorizer"] = vectorizer
    blocks = [sparse.csr_matrix(preprocess["scaler"].transform(image))]
    if mode != "image_only":
        blocks.append(preprocess["vectorizer"].transform(get_text(frame, root)))
    if mode == "2b_image_text_gold":
        blocks.append(sparse.csr_matrix(multilabel(frame, "text_labels", labels).astype(float)))
    return sparse.hstack(blocks, format="csr"), preprocess


def fit_heads(x: sparse.csr_matrix, y: np.ndarray, fit: np.ndarray, seed: int) -> list[object]:
    heads = []
    for column in range(y.shape[1]):
        target = y[fit, column]
        if len(np.unique(target)) == 1:
            heads.append(float(target[0]))
        else:
            head = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            head.fit(x[fit], target)
            heads.append(head)
    return heads


def probabilities(heads: list[object], x: sparse.csr_matrix) -> np.ndarray:
    return np.column_stack([np.full(x.shape[0], head) if isinstance(head, float)
                            else head.predict_proba(x)[:, 1] for head in heads])


def metric_bundle(y: np.ndarray, pred: np.ndarray, labels: list[str], none_mode: str) -> dict:
    if none_mode == "include":
        y = np.column_stack([y, (y.sum(axis=1) == 0).astype(int)])
        pred = np.column_stack([pred, (pred.sum(axis=1) == 0).astype(int)])
        labels = labels + ["None"]
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    return {
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "tp": int(np.logical_and(y, pred).sum()),
        "fp": int(np.logical_and(1 - y, pred).sum()),
        "fn": int(np.logical_and(y, 1 - pred).sum()),
        "per_label": {label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for label, p, r, f, s in zip(labels, precision, recall, f1, support)},
        "zero_label_errors": int(np.logical_and(y[:, :len(labels) - (none_mode == "include")].sum(axis=1) == 0,
                                                pred[:, :len(labels) - (none_mode == "include")].sum(axis=1) > 0).sum()),
    }


def calibrate(y: np.ndarray, p: np.ndarray, labels: list[str], none_mode: str) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    return float(max(candidates, key=lambda threshold: (metric_bundle(y, (p >= threshold).astype(np.uint8), labels, none_mode)["micro_f1"], -abs(threshold - 0.5))))


def save_predictions(path: Path, frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, pred: np.ndarray, labels: list[str]) -> None:
    result = frame[["image", "language", "group"]].copy()
    result["gold"] = ["|".join(np.asarray(labels)[row.astype(bool)]) for row in y]
    result["predicted"] = ["|".join(np.asarray(labels)[row.astype(bool)]) for row in pred]
    result["probabilities"] = [json.dumps(dict(zip(labels, map(float, row)))) for row in p]
    result.to_csv(path, index=False)


def train(frame: pd.DataFrame, root: Path, labels: list[str], out: Path, kind: str,
          checkpoint: str | None, enable_2b: bool, seed: int, none_mode: str) -> None:
    train_frame = frame.loc[frame.split == "train"].reset_index(drop=True)
    if set(train_frame.group) & set(frame.loc[frame.split != "train", "group"]):
        raise ValueError("Train group overlaps dev/test; resolve leakage before training")
    if len(train_frame) < 3:
        raise ValueError("At least three labeled training examples are needed")
    y = multilabel(train_frame, "image_labels", labels)
    assignments = split_groups(train_frame, y, seed)
    fit = np.flatnonzero(assignments == "fit")
    calibration = np.flatnonzero(assignments == "calibration")
    evaluation = np.flatnonzero(assignments == "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = train_frame.copy()
    snapshot["internal_split"] = assignments
    snapshot.to_csv(out / "split_manifest.csv", index=False)
    (out / "versions.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                                                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                                     "scipy": scipy.__version__}, indent=2))
    image = image_features(train_frame, root, kind, checkpoint)
    modes = ["image_only", "2a_image_text"] + (["2b_image_text_gold"] if enable_2b else [])
    for mode in modes:
        folder = out / mode
        folder.mkdir(exist_ok=True)
        x, prep = build_matrix(train_frame, root, image, labels, mode, fit_indices=fit)
        heads = fit_heads(x, y, fit, seed)
        p_cal = probabilities(heads, x[calibration])
        threshold = calibrate(y[calibration], p_cal, labels, none_mode)
        p_eval = probabilities(heads, x[evaluation])
        pred_eval = (p_eval >= threshold).astype(np.uint8)
        metrics = metric_bundle(y[evaluation], pred_eval, labels, none_mode)
        metrics.update({"threshold": threshold, "fit_rows": len(fit), "calibration_rows": len(calibration),
                        "evaluation_rows": len(evaluation), "constant_heads": [labels[i] for i, h in enumerate(heads) if isinstance(h, float)],
                        "none_mode": none_mode, "model": kind, "mode": mode})
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (folder / "config.json").write_text(json.dumps({"model": kind, "mode": mode, "checkpoint": checkpoint,
                                                        "seed": seed, "none_mode": none_mode, "labels": labels}, indent=2))
        save_predictions(folder / "evaluation_predictions.csv", train_frame.iloc[evaluation], y[evaluation], p_eval, pred_eval, labels)
        joblib.dump({"preprocess": prep, "heads": heads, "threshold": threshold, "labels": labels,
                     "mode": mode, "kind": kind, "checkpoint": checkpoint}, folder / "model.joblib")


def write_response(run: Path, frame: pd.DataFrame, root: Path, labels: list[str], split: str, out: Path) -> None:
    model = joblib.load(run / "model.joblib")
    if model["labels"] != labels:
        raise ValueError("Label file differs from trained model")
    subset = frame.loc[frame.split == split].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {split} rows")
    image = image_features(subset, root, model["kind"], model["checkpoint"])
    x, _ = build_matrix(subset, root, image, labels, model["mode"], preprocess=model["preprocess"])
    pred = probabilities(model["heads"], x) >= model["threshold"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        for image_name, row in zip(subset.image, pred):
            # A trailing tab denotes a known zero-label prediction.
            stream.write(image_name + "\t" + "\t".join(np.asarray(labels)[row]) + "\n")
