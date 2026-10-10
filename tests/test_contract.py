import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from framing.data import read_manifest
from framing.model import metric_bundle, split_groups
from framing.scoring import score_tsv


class ContractTests(unittest.TestCase):
    def test_group_split_keeps_articles_together(self):
        frame = pd.DataFrame({"group": [f"article-{i // 2}" for i in range(40)],
                              "language": ["EN", "BG", "PT", "RU"] * 10})
        y = np.asarray([[i % 3 == 0, i % 4 == 0] for i in range(40)], dtype=np.uint8)
        assigned = split_groups(frame, y, 42)
        self.assertEqual(set(assigned), {"fit", "calibration", "evaluation"})
        self.assertTrue(all(len(set(assigned[frame.group == group])) == 1 for group in frame.group.unique()))

    def test_empty_prediction_counts_false_positive_and_none_choice(self):
        y = np.asarray([[0, 0], [1, 0]], dtype=np.uint8)
        pred = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
        plain = metric_bundle(y, pred, ["A", "B"], "exclude")
        with_none = metric_bundle(y, pred, ["A", "B"], "include")
        self.assertEqual((plain["tp"], plain["fp"], plain["fn"]), (1, 1, 0))
        self.assertGreater(with_none["fn"], plain["fn"])

    def test_tsv_requires_exact_image_set_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, response = root / "gold.tsv", root / "response.tsv"
            gold.write_text("a.png\t\nb.png\tA\n")
            response.write_text("a.png\t\nb.png\tA\n")
            self.assertEqual(score_tsv(gold, response, ["A"], "exclude")["micro_f1"], 1.0)
            response.write_text("a.png\t\nb.png\tBad\n")
            with self.assertRaises(ValueError):
                score_tsv(gold, response, ["A"], "exclude")

    def test_manifest_rejects_unknown_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("image,article,language,split,image_labels\na.png,a.txt,EN,train,Unknown\n")
            with self.assertRaises(ValueError):
                read_manifest(path, ["Known"])


if __name__ == "__main__":
    unittest.main()
