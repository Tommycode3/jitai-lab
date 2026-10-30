#!/usr/bin/env bash
# Reproduce every number and figure in the README. Single-threaded, sequential.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$PY" study_coverage.py
"$PY" study_moderation.py
"$PY" study_power.py
"$PY" study_thompson.py
"$PY" make_figures.py
