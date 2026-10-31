#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# != 2 )); then
    echo "Usage: bash scripts/average_checkpoints.sh CHECKPOINT_DIR OUTPUT.pt" >&2
    exit 2
fi
python "$ROOT/third_party/fairseq/scripts/average_checkpoints.py" \
    --inputs "$1" --output "$2" --num-update-checkpoints 5
