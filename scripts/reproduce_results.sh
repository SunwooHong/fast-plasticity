#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/reproduce_results.py
python scripts/make_figures.py
