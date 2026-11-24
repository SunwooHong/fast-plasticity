#!/usr/bin/env bash
# Copy tracked release files into a clean directory. No .git, caches, or checkpoints.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:?destination directory}"
mkdir -p "$dest"
cd "$root"
git ls-files -z | while IFS= read -r -d '' path; do
  case "$path" in
    *.pyc|*/__pycache__/*) continue ;;
  esac
  mkdir -p "$dest/$(dirname "$path")"
  cp -a "$path" "$dest/$path"
done
find "$dest" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$dest" -name '*.pyc' -delete
echo "wrote $dest"
