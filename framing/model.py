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


def split_groups(frame: pd.DataFrame, targets: np.ndarray, seed: int) -> np.ndarray:
    """Find a grouped split with similar label and language rates in each part."""
    if frame.group.nunique() < 3:
        raise ValueError("At least three independent groups are needed for fit, calibration, evaluation")
    best_split = None
    overall_label_rate = targets.mean(axis=0)
    language_indicators = pd.get_dummies(frame.language).to_numpy(dtype=float)
    overall_language_rate = language_indicators.mean(axis=0)
    for candidate in range(100):
        try:
            first_split = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + candidate)
            fit_rows, remaining_rows = next(first_split.split(frame, groups=frame.group))
            second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1000 + candidate)
            calibration_local, evaluation_local = next(second_split.split(frame.iloc[remaining_rows], groups=frame.group.iloc[remaining_rows]))
        except ValueError:
            continue
        candidate_parts = [fit_rows, remaining_rows[calibration_local], remaining_rows[evaluation_local]]
        if min(map(len, candidate_parts)) == 0:
            continue
        balance_cost = sum(
            np.abs(targets[rows].mean(axis=0) - overall_label_rate).mean()
            + np.abs(language_indicators[rows].mean(axis=0) - overall_language_rate).mean()
            for rows in candidate_parts
        )
        if best_split is None or balance_cost < best_split[0]:
            best_split = balance_cost, candidate_parts
    if best_split is None:
        raise ValueError("Could not create three nonempty grouped partitions")
    split_names = np.empty(len(frame), dtype=object)
    for rows, name in zip(best_split[1], ["fit", "calibration", "evaluation"]):
        split_names[rows] = name
    return split_names


def frozen_features(paths: list[Path], checkpoint: str, batch_size: int = 16) -> np.ndarray:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Frozen baseline needs requirements-pretrained.txt") from error
    from .data import image_rgb
    if checkpoint == "torchvision:resnet18":
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise RuntimeError("ResNet18 needs torchvision from requirements-pretrained.txt") from error
        weights = ResNet18_Weights.IMAGENET1K_V1
        processor = weights.transforms()
        model = resnet18(weights=weights).eval()
        model.fc = torch.nn.Identity()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding_batches = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = torch.stack([processor(image_rgb(path)) for path in paths[start:start + batch_size]])
                embedding_batches.append(model(images).float().cpu().numpy())
        return np.vstack(embedding_batches).astype(np.float32)
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("Hugging Face checkpoint needs transformers from requirements-pretrained.txt") from error
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding_batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = [image_rgb(path) for path in paths[start:start + batch_size]]
            inputs = processor(images=images, return_tensors="pt")
            if hasattr(model, "get_image_features"):
                embeddings = model.get_image_features(**inputs)
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
            else:
                output = model(**inputs)
                embeddings = output.pooler_output if output.pooler_output is not None else output.last_hidden_state.mean(dim=1)
            embedding_batches.append(embeddings.float().cpu().numpy())
    return np.vstack(embedding_batches).astype(np.float32)


def image_features(frame: pd.DataFrame, root: Path, kind: str, checkpoint: str | None) -> np.ndarray:
    image_paths = [resolve(root, value) for value in frame.image_path]
    if kind == "classical":
        return np.vstack([classical_features(path) for path in image_paths])
    if not checkpoint:
        raise ValueError("--checkpoint is required for frozen model")
    return frozen_features(image_paths, checkpoint)


def get_text(frame: pd.DataFrame, root: Path) -> list[str]:
    return [article_text(resolve(root, value)) for value in frame.article]


def build_matrix(frame: pd.DataFrame, root: Path, image_vectors: np.ndarray, labels: list[str], mode: str,
                 fit_indices: np.ndarray | None = None, preprocess: dict | None = None) -> tuple[sparse.csr_matrix, dict]:
    if preprocess is None:
        scaler = StandardScaler().fit(image_vectors[fit_indices])
        preprocess = {"scaler": scaler, "vectorizer": None}
        if mode != "image_only":
            # fit the article vocabulary on the fit rows only.
            vectorizer = TfidfVectorizer(max_features=20000, min_df=1, ngram_range=(1, 2), sublinear_tf=True)
            vectorizer.fit([get_text(frame.iloc[[row_number]], root)[0] for row_number in fit_indices])
            preprocess["vectorizer"] = vectorizer
    feature_blocks = [sparse.csr_matrix(preprocess["scaler"].transform(image_vectors))]
    if mode != "image_only":
        feature_blocks.append(preprocess["vectorizer"].transform(get_text(frame, root)))
    if mode == "2b_image_text_gold":
        feature_blocks.append(sparse.csr_matrix(multilabel(frame, "text_labels", labels).astype(float)))
    return sparse.hstack(feature_blocks, format="csr"), preprocess


def fit_heads(feature_matrix: sparse.csr_matrix, targets: np.ndarray, fit_rows: np.ndarray, seed: int) -> list[object]:
    heads = []
    for label_column in range(targets.shape[1]):
        label_targets = targets[fit_rows, label_column]
        if len(np.unique(label_targets)) == 1:
            heads.append(float(label_targets[0]))
        else:
            head = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            head.fit(feature_matrix[fit_rows], label_targets)
            heads.append(head)
    return heads


def probabilities(heads: list[object], feature_matrix: sparse.csr_matrix) -> np.ndarray:
    return np.column_stack([
        np.full(feature_matrix.shape[0], head) if isinstance(head, float)
        else head.predict_proba(feature_matrix)[:, 1]
        for head in heads
    ])


def metric_bundle(targets: np.ndarray, predictions: np.ndarray, labels: list[str], none_mode: str) -> dict:
    image_label_count = len(labels)
    if none_mode == "include":
        targets = np.column_stack([targets, (targets.sum(axis=1) == 0).astype(int)])
        predictions = np.column_stack([predictions, (predictions.sum(axis=1) == 0).astype(int)])
        labels = labels + ["None"]
    precision, recall, label_f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    return {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "tp": int(np.logical_and(targets, predictions).sum()),
        "fp": int(np.logical_and(1 - targets, predictions).sum()),
        "fn": int(np.logical_and(targets, 1 - predictions).sum()),
        "per_label": {
            label: {"precision": float(label_precision), "recall": float(label_recall),
                    "f1": float(frame_f1), "support": int(label_support)}
            for label, label_precision, label_recall, frame_f1, label_support
            in zip(labels, precision, recall, label_f1, support)
        },
        "zero_label_errors": int(np.logical_and(
            targets[:, :image_label_count].sum(axis=1) == 0,
            predictions[:, :image_label_count].sum(axis=1) > 0,
        ).sum()),
    }


def calibrate(targets: np.ndarray, probabilities_by_label: np.ndarray, labels: list[str], none_mode: str) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, 17)
    return float(max(
        candidate_thresholds,
        key=lambda threshold: (
            metric_bundle(targets, (probabilities_by_label >= threshold).astype(np.uint8), labels, none_mode)["micro_f1"],
            -abs(threshold - 0.5),
        ),
    ))


def save_predictions(path: Path, frame: pd.DataFrame, targets: np.ndarray,
                     probabilities_by_label: np.ndarray, predictions: np.ndarray, labels: list[str]) -> None:
    result = frame[["image", "language", "group"]].copy()
    result["gold"] = ["|".join(np.asarray(labels)[row.astype(bool)]) for row in targets]
    result["predicted"] = ["|".join(np.asarray(labels)[row.astype(bool)]) for row in predictions]
    result["probabilities"] = [
        json.dumps(dict(zip(labels, map(float, row)))) for row in probabilities_by_label
    ]
    result.to_csv(path, index=False)


def train(frame: pd.DataFrame, root: Path, labels: list[str], output_dir: Path, kind: str,
          checkpoint: str | None, enable_2b: bool, seed: int, none_mode: str) -> None:
    training_examples = frame.loc[frame.split == "train"].reset_index(drop=True)
    if set(training_examples.group) & set(frame.loc[frame.split != "train", "group"]):
        raise ValueError("Train group overlaps dev/test; resolve leakage before training")
    if len(training_examples) < 3:
        raise ValueError("At least three labeled training examples are needed")
    targets = multilabel(training_examples, "image_labels", labels)
    split_assignments = split_groups(training_examples, targets, seed)
    fit_rows = np.flatnonzero(split_assignments == "fit")
    calibration_rows = np.flatnonzero(split_assignments == "calibration")
    evaluation_rows = np.flatnonzero(split_assignments == "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = training_examples.copy()
    split_manifest["internal_split"] = split_assignments
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__}
    if kind == "frozen":
        import torch
        import torchvision
        versions.update({"torch": torch.__version__, "torchvision": torchvision.__version__})
    (output_dir / "versions.json").write_text(json.dumps(versions, indent=2))
    image_vectors = image_features(training_examples, root, kind, checkpoint)
    modes = ["image_only", "2a_image_text"] + (["2b_image_text_gold"] if enable_2b else [])
    for mode in modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        feature_matrix, preprocessing = build_matrix(
            training_examples, root, image_vectors, labels, mode, fit_indices=fit_rows
        )
        heads = fit_heads(feature_matrix, targets, fit_rows, seed)
        calibration_probabilities = probabilities(heads, feature_matrix[calibration_rows])
        threshold = calibrate(targets[calibration_rows], calibration_probabilities, labels, none_mode)
        evaluation_probabilities = probabilities(heads, feature_matrix[evaluation_rows])
        evaluation_predictions = (evaluation_probabilities >= threshold).astype(np.uint8)
        metrics = metric_bundle(targets[evaluation_rows], evaluation_predictions, labels, none_mode)
        metrics.update({"threshold": threshold, "fit_rows": len(fit_rows), "calibration_rows": len(calibration_rows),
                        "evaluation_rows": len(evaluation_rows),
                        "constant_heads": [labels[label_index] for label_index, head in enumerate(heads) if isinstance(head, float)],
                        "none_mode": none_mode, "model": kind, "mode": mode})
        (mode_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (mode_dir / "config.json").write_text(json.dumps({"model": kind, "mode": mode, "checkpoint": checkpoint,
                                                        "seed": seed, "none_mode": none_mode, "labels": labels}, indent=2))
        save_predictions(mode_dir / "evaluation_predictions.csv", training_examples.iloc[evaluation_rows],
                         targets[evaluation_rows], evaluation_probabilities, evaluation_predictions, labels)
        joblib.dump({"preprocess": preprocessing, "heads": heads, "threshold": threshold, "labels": labels,
                     "mode": mode, "kind": kind, "checkpoint": checkpoint}, mode_dir / "model.joblib")


def write_response(run: Path, frame: pd.DataFrame, root: Path, labels: list[str], split: str, output_path: Path) -> None:
    saved_model = joblib.load(run / "model.joblib")
    if saved_model["labels"] != labels:
        raise ValueError("Label file differs from trained model")
    selected_examples = frame.loc[frame.split == split].reset_index(drop=True)
    if selected_examples.empty:
        raise ValueError(f"No {split} rows")
    image_vectors = image_features(selected_examples, root, saved_model["kind"], saved_model["checkpoint"])
    feature_matrix, _ = build_matrix(
        selected_examples, root, image_vectors, labels, saved_model["mode"], preprocess=saved_model["preprocess"]
    )
    predicted_labels = probabilities(saved_model["heads"], feature_matrix) >= saved_model["threshold"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        for image_name, label_row in zip(selected_examples.image, predicted_labels):
            # keep the empty field when the model finds no frame.
            stream.write(image_name + "\t" + "\t".join(np.asarray(labels)[label_row]) + "\n")
