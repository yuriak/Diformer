#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: bash scripts/preprocess.sh SOURCE TARGET TEXT_DIR DEST_DIR [fairseq-preprocess options]" >&2
    exit 2
fi
SRC=$1
TGT=$2
TEXT=$3
DEST=$4
shift 4
# Input files: train.SOURCE, train.TARGET, valid.*, test.*; already Moses/BPE encoded.
fairseq-preprocess --source-lang "$SRC" --target-lang "$TGT" \
    --trainpref "$TEXT/train" --validpref "$TEXT/valid" --testpref "$TEXT/test" \
    --destdir "$DEST" --joined-dictionary --workers "${WORKERS:-4}" "$@"
