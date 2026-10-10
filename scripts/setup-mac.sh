#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-/Users/abdulmoiz/Documents/Semeval Task9}"
repository_url="https://github.com/moizart/semeval-task9-2027.git"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required. Install Git, then run this script again." >&2
  exit 1
fi
python_cmd=""
for candidate in python3.12 python3.11 python3.13 python3; do
  if ! command -v "$candidate" >/dev/null 2>&1; then
    continue
  fi
  if ! "$candidate" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))'; then
    continue
  fi
  # Check venv and pip before changing the project. Some Python builds omit ensurepip.
  probe_dir="$(mktemp -d)"
  if "$candidate" -m venv "$probe_dir/venv" >/dev/null 2>&1 &&
      "$probe_dir/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    python_cmd="$(command -v "$candidate")"
    rm -rf "$probe_dir"
    break
  fi
  rm -rf "$probe_dir"
done
if [ -z "$python_cmd" ]; then
  echo "No working Python 3.11-3.13 with venv and pip was found." >&2
  echo "Install Python 3.12 (for example: brew install python@3.12), then rerun this script." >&2
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

if [ -e "$project_dir/.venv" ]; then
  if [ ! -f "$project_dir/.venv/pyvenv.cfg" ]; then
    echo "$project_dir/.venv exists but is not a Python virtual environment. Nothing was changed." >&2
    exit 1
  fi
  # Rebuild a partial environment left by a failed ensurepip invocation.
  "$python_cmd" -m venv --clear "$project_dir/.venv"
else
  "$python_cmd" -m venv "$project_dir/.venv"
fi
"$project_dir/.venv/bin/python" -m pip install -r "$project_dir/requirements.txt"
(
  cd "$project_dir"
  .venv/bin/python -m unittest discover -s tests -v
  mkdir -p data runs
)

echo
echo "Project ready at: $project_dir"
echo "Base Python: $python_cmd"
echo "Python: $project_dir/.venv/bin/python"
echo "Next: add organizer data and data/manifest.csv, then follow README.md."
