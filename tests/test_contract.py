import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from framing.data import read_manifest
from framing.model import metric_bundle, split_groups
from framing.scoring import score_tsv
from framing.trial import prepare_trial
from framing.release import prepare_release


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

    def test_trial_gold_accepts_filename_only_for_unframed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, response = root / "gold.tsv", root / "response.tsv"
            gold.write_text("a.png\nb.png\tA\n")
            response.write_text("a.png\t\nb.png\tA\n")
            self.assertEqual(score_tsv(gold, response, ["A"], "include")["micro_f1"], 1.0)

    def test_trial_import_joins_image_article_and_text_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "trial.zip"
            labels = ["Political"] + [f"Other_{i}" for i in range(13)]
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("labels.txt", "\n".join(labels))
                for language in ("BG", "EN", "PT", "RU"):
                    name = f"{language}_1"
                    bundle.writestr(f"{language}/raw-documents/{name}.png", b"image")
                    bundle.writestr(f"{language}/raw-documents/{name}.txt", b"article")
                    bundle.writestr(f"{language}/subtask-1-annotations.txt", f"{name}.txt\t0\t7\tPolitical\n")
                    gold = f"{name}.png" if language == "EN" else f"{name}.png\tPolitical"
                    bundle.writestr(f"{language}/subtask-2-annotations.txt", gold + "\n")
            result = prepare_trial(archive, root / "data")
            manifest = pd.read_csv(root / "data/manifest.csv").fillna("")
            self.assertEqual(result["examples"], 4)
            self.assertEqual(manifest.loc[manifest.language == "EN", "image_labels"].iloc[0], "")
            self.assertTrue((manifest.text_labels == "Political").all())

    def test_attached_release_imports_400_pairs_without_inventing_text_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "release.zip"
            labels = ["Political"] + [f"Other_{i}" for i in range(13)]
            with ZipFile(archive, "w") as bundle:
                vocabulary = "\n".join(labels) + "\n"
                for release in ("A", "B"):
                    bundle.writestr(f"data/subtask1/SEMEVAL23_FRAMING_INITIAL_TRAIN-{release}/labels.txt", vocabulary)
                lines = []
                for number in range(1, 401):
                    bundle.writestr(f"data/subtask2/images/EN_doc_{number}_image.jpg", b"image")
                    bundle.writestr(f"data/subtask2/images/EN_M_TRAIN_doc_{number}.txt", b"article")
                    label = "\tPolitical" if number % 2 else ""
                    lines.append(f"EN_doc_{number}_image.jpg{label}")
                bundle.writestr("data/subtask2/subtask-2-annotations.txt", "\n".join(lines) + "\n")
            result = prepare_release(archive, root / "data")
            manifest = pd.read_csv(root / "data/manifest.csv").fillna("")
            self.assertEqual(result["examples"], 400)
            self.assertEqual(len(manifest), 400)
            self.assertNotIn("text_labels", manifest.columns)
            self.assertEqual(manifest.loc[0, "article"], "extracted/data/subtask2/images/EN_M_TRAIN_doc_1.txt")
            self.assertEqual(manifest.loc[1, "image_labels"], "")

    def test_manifest_rejects_unknown_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("image,article,language,split,image_labels\na.png,a.txt,EN,train,Unknown\n")
            with self.assertRaises(ValueError):
                read_manifest(path, ["Known"])


if __name__ == "__main__":
    unittest.main()
