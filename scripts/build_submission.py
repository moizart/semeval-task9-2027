"""Build one self-contained Assignment 1 ZIP from the reviewed project files."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "Assignment_1_Group_12"


def submission_files() -> list[Path]:
    """List sources and evidence explicitly so local run files stay outside the ZIP."""
    relative_files = [
        Path("Agent.md"), Path("Changes.md"), Path("PLAN.md"), Path("PROJECT_GUIDE.md"),
        Path("requirements.txt"), Path("requirements-pretrained.txt"),
        Path("scripts/setup-mac.sh"), Path("scripts/build_submission.py"),
        Path("submission/README.md"), Path("data/README.md"),
        Path("report/main.tex"), Path("report/references.bib"),
    ]
    relative_files.extend(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "framing").glob("*.py"))
    relative_files.extend(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "tests").glob("*.py"))
    relative_files.extend(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "artifacts").iterdir() if path.is_file())
    relative_files.extend(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "data/release1").iterdir() if path.is_file())
    for relative_path in relative_files:
        if not (PROJECT_ROOT / relative_path).is_file():
            raise FileNotFoundError(f"Submission input is missing: {relative_path}")
    return sorted(set(relative_files))


def compile_report(build_dir: Path) -> Path:
    """Build the PDF with BibTeX in a temporary folder, away from the editor."""
    if shutil.which("latexmk") is None:
        raise RuntimeError("latexmk is needed to build the submission PDF")
    report_dir = build_dir / "report"
    report_dir.mkdir()
    for name in ("main.tex", "references.bib"):
        shutil.copy2(PROJECT_ROOT / "report" / name, report_dir / name)
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    result = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("Report build failed:\n" + (result.stdout + result.stderr)[-5000:])
    report_pdf = report_dir / "main.pdf"
    if not report_pdf.is_file():
        raise RuntimeError("latexmk finished without a report PDF")
    return report_pdf


def build_submission() -> Path:
    output_dir = PROJECT_ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    package_path = output_dir / f"{PACKAGE_NAME}.zip"
    with tempfile.TemporaryDirectory(prefix="assignment1-report-") as temporary_dir:
        report_pdf = compile_report(Path(temporary_dir))
        shutil.copy2(report_pdf, output_dir / "Assignment_1_Report_Group_12.pdf")
        with ZipFile(package_path, "w") as package:
            for relative_path in submission_files():
                compression = ZIP_STORED if relative_path == Path("data/release1/data.zip") else ZIP_DEFLATED
                package.write(PROJECT_ROOT / relative_path, f"{PACKAGE_NAME}/{relative_path.as_posix()}", compress_type=compression)
            package.write(PROJECT_ROOT / "submission/README.md", f"{PACKAGE_NAME}/README.md", compress_type=ZIP_DEFLATED)
            package.write(PROJECT_ROOT / "README.md", f"{PACKAGE_NAME}/docs/REPOSITORY_README.md", compress_type=ZIP_DEFLATED)
            package.write(report_pdf, f"{PACKAGE_NAME}/report/Assignment_1_Report_Group_12.pdf", compress_type=ZIP_DEFLATED)
    with ZipFile(package_path) as package:
        damaged_file = package.testzip()
        if damaged_file is not None:
            raise RuntimeError(f"Submission ZIP failed integrity check: {damaged_file}")
    digest = hashlib.sha256()
    with package_path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_hash = digest.hexdigest()
    print(f"Built {package_path} ({package_path.stat().st_size:,} bytes)")
    print(f"SHA-256: {archive_hash}")
    return package_path


if __name__ == "__main__":
    build_submission()
