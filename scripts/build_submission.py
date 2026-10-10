"""Package the Assignment 1 deliverables and the code needed to reproduce them."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "Assignment_1_Group_12"


def submission_files() -> dict[Path, Path]:
    """Map reviewed project files to their paths inside the submission."""
    files = {
        Path("submission/README.md"): Path("README.md"),
        Path("report/A1_Semeval_Task9_Group12_Report.pdf"): Path("A1_Semeval_Task9_Group12_Report.pdf"),
        Path("requirements.txt"): Path("requirements.txt"),
        Path("requirements-pretrained.txt"): Path("requirements-pretrained.txt"),
        Path("data/release1/data.zip"): Path("data/release1/data.zip"),
        Path("artifacts/release1-eda.pdf"): Path("artifacts/release1-eda.pdf"),
        Path("artifacts/release1-metrics.csv"): Path("artifacts/release1-metrics.csv"),
        Path("artifacts/release1-audit.json"): Path("artifacts/release1-audit.json"),
        Path("artifacts/release1-split-manifest.csv"): Path("artifacts/release1-split-manifest.csv"),
        Path("tests/test_contract.py"): Path("tests/test_contract.py"),
    }
    for source in sorted((PROJECT_ROOT / "framing").glob("*.py")):
        relative_path = source.relative_to(PROJECT_ROOT)
        files[relative_path] = relative_path
    for relative_path in files:
        if not (PROJECT_ROOT / relative_path).is_file():
            raise FileNotFoundError(f"Submission input is missing: {relative_path}")
    return files


def build_submission() -> Path:
    report = PROJECT_ROOT / "report/A1_Semeval_Task9_Group12_Report.pdf"
    with report.open("rb") as report_file:
        if report_file.read(4) != b"%PDF":
            raise ValueError("The exported report is not a PDF")
    output_dir = PROJECT_ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    package_path = output_dir / f"{PACKAGE_NAME}.zip"
    with ZipFile(package_path, "w") as package:
        for source_path, archive_path in submission_files().items():
            compression = ZIP_STORED if source_path == Path("data/release1/data.zip") else ZIP_DEFLATED
            package.write(PROJECT_ROOT / source_path, f"{PACKAGE_NAME}/{archive_path.as_posix()}", compress_type=compression)
    with ZipFile(package_path) as package:
        damaged_file = package.testzip()
        if damaged_file:
            raise RuntimeError(f"Submission ZIP failed integrity check: {damaged_file}")
    digest = hashlib.sha256()
    with package_path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"Built {package_path} ({package_path.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest.hexdigest()}")
    return package_path


if __name__ == "__main__":
    build_submission()
