from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_labels, read_manifest
from .model import train, write_response
from .scoring import score_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="SemEval Task 9 Subtask 2 baseline workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "eda", "train", "predict"):
        item = sub.add_parser(name)
        item.add_argument("--manifest", required=True, type=Path)
        item.add_argument("--labels", required=True, type=Path)
        item.add_argument("--data-root", type=Path, help="Root of paths in manifest; defaults to manifest directory")
        if name in ("audit", "eda", "train", "predict"):
            item.add_argument("--out", required=True, type=Path)
        if name == "train":
            item.add_argument("--model", choices=["classical", "frozen"], default="classical")
            item.add_argument("--checkpoint", type=str)
            item.add_argument("--enable-2b", action="store_true", help="Use gold text labels supplied for every row")
            item.add_argument("--seed", type=int, default=42)
            item.add_argument("--none-mode", choices=["exclude", "include"], default="exclude")
        if name == "predict":
            item.add_argument("--run", required=True, type=Path)
            item.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    score = sub.add_parser("score")
    score.add_argument("--gold", required=True, type=Path)
    score.add_argument("--response", required=True, type=Path)
    score.add_argument("--labels", required=True, type=Path)
    score.add_argument("--none-mode", choices=["exclude", "include"], default="exclude")
    args = parser.parse_args()
    labels = read_labels(args.labels)
    if args.command == "score":
        print(json.dumps(score_tsv(args.gold, args.response, labels, args.none_mode), indent=2))
        return
    frame = read_manifest(args.manifest, labels)
    root = args.data_root or args.manifest.parent
    if args.command == "audit":
        from .audit import save_audit
        _, report = save_audit(frame, root, labels, args.out)
        print(json.dumps({"rows": len(frame), "errors": len(report["errors"]), "out": str(args.out)}))
    elif args.command == "eda":
        from .audit import save_audit, save_eda
        stats, report = save_audit(frame, root, labels, args.out)
        if report["errors"]:
            raise ValueError(f"Audit found {len(report['errors'])} missing or corrupt files; see {args.out / 'audit.json'}")
        save_eda(frame, stats, labels, args.out)
        print(args.out / "eda.pdf")
    elif args.command == "train":
        if args.enable_2b and "text_labels" not in __import__("pandas").read_csv(args.manifest, nrows=0).columns:
            raise ValueError("--enable-2b needs a text_labels column")
        train(frame, root, labels, args.out, args.model, args.checkpoint, args.enable_2b, args.seed, args.none_mode)
        print(args.out)
    elif args.command == "predict":
        write_response(args.run, frame, root, labels, args.split, args.out)
        print(args.out)


if __name__ == "__main__":
    main()
