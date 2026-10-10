from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import ImageFilter

from .data import image_rgb


FEATURE_NAMES = [
    "log_width", "log_height", "log_aspect", "log_pixels", "brightness_mean",
    "brightness_std", "saturation_mean", "saturation_std", "edge_density",
    "edge_strength", "top_brightness", "bottom_brightness", "center_brightness",
    "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std",
] + [f"gray_hist_{bin_index}" for bin_index in range(16)] + [f"color_hist_{channel}_{bin_index}" for channel in "rgb" for bin_index in range(8)]


def classical_features(path: Path) -> np.ndarray:
    image = image_rgb(path)
    width, height = image.size
    image.thumbnail((256, 256))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    gray = pixels @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    high = pixels.max(axis=2)
    low = pixels.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-8), 0)
    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    row_count, col_count = gray.shape
    top = gray[:max(1, row_count // 3)]
    bottom = gray[-max(1, row_count // 3):]
    center = gray[row_count // 4:max(row_count // 4 + 1, 3 * row_count // 4), col_count // 4:max(col_count // 4 + 1, 3 * col_count // 4)]
    values = [
        np.log1p(width), np.log1p(height), np.log(width / height), np.log1p(width * height),
        gray.mean(), gray.std(), saturation.mean(), saturation.std(),
        (edges > 0.2).mean(), edges.mean(), top.mean(), bottom.mean(), center.mean(),
    ]
    values.extend(pixels.mean(axis=(0, 1)))
    values.extend(pixels.std(axis=(0, 1)))
    values.extend(np.histogram(gray, bins=16, range=(0, 1), density=False)[0] / gray.size)
    for channel in range(3):
        values.extend(np.histogram(pixels[:, :, channel], bins=8, range=(0, 1), density=False)[0] / gray.size)
    return np.asarray(values, dtype=np.float32)
