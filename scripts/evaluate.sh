#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 2 )); then
    echo "Usage: bash scripts/evaluate.sh GENERATE_FILE TOKENIZED_REFERENCE [sacrebleu options]" >&2
    exit 2
fi
GENERATION=$1
REFERENCE=$2
shift 2
python "$ROOT/scripts/extract_hypotheses.py" "$GENERATION" | \
    python -m sacrebleu "$REFERENCE" --tokenize none "$@"
