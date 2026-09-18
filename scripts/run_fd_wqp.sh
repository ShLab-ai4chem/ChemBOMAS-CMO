#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src" \
  python -m chembomas.wet.fd_wqp.run "$@"
