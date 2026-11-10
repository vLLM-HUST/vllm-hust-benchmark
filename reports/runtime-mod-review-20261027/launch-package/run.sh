#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p evidence
if [[ -e evidence/gamma4-campaign-status.json ]]; then
  echo "Existing campaign evidence found; use a fresh extracted directory." >&2
  exit 1
fi
exec venv/bin/python -u campaign_gamma4.py
