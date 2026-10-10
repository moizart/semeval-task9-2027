#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-/Users/abdulmoiz/Documents/Semeval Task9}"
repository_url="https://github.com/moizart/semeval-task9-2027.git"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required. Install Git, then run this script again." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install Python 3.11 or newer, then run this script again." >&2
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Python 3.11 or newer is required. Current version: $(python3 --version)" >&2
  exit 1
fi

if [ -d "$project_dir/.git" ]; then
  current_origin="$(git -C "$project_dir" remote get-url origin 2>/dev/null || true)"
  case "$current_origin" in
    *moizart/semeval-task9-2027.git) ;;
    *) echo "The existing Git repository at $project_dir has a different origin. Nothing was changed." >&2; exit 1 ;;
  esac
  git -C "$project_dir" fetch origin work
  git -C "$project_dir" switch work
  git -C "$project_dir" pull --ff-only origin work
elif [ -e "$project_dir" ] && [ ! -d "$project_dir" ]; then
  echo "$project_dir is a file. Nothing was changed." >&2
  exit 1
elif [ -d "$project_dir" ] && [ -n "$(ls -A "$project_dir")" ]; then
  echo "$project_dir contains files but is not this Git repository. Nothing was changed." >&2
  echo "Move those files to a safe location or pass an empty target directory as the first argument." >&2
  exit 1
else
  mkdir -p "$(dirname "$project_dir")"
  git clone --branch work --single-branch "$repository_url" "$project_dir"
fi

python3 -m venv "$project_dir/.venv"
"$project_dir/.venv/bin/python" -m pip install -r "$project_dir/requirements.txt"
(
  cd "$project_dir"
  .venv/bin/python -m unittest discover -s tests -v
  mkdir -p data runs
)

echo
echo "Project ready at: $project_dir"
echo "Python: $project_dir/.venv/bin/python"
echo "Next: add organizer data and data/manifest.csv, then follow README.md."
