#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src" \
  python -m chembomas.dry.edbo.run "$@"
