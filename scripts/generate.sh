#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 6 )); then
    echo "Usage: bash scripts/generate.sh DATA_BIN CHECKPOINT SOURCE TARGET {l2r|r2l|mask_predict|easy_first} OUTPUT_DIR [fairseq-generate options]" >&2
    exit 2
fi
DATA=$1
CHECKPOINT=$2
SRC=$3
TGT=$4
STRATEGY=$5
OUTPUT=$6
shift 6
case "$STRATEGY" in
    l2r|r2l|mask_predict|easy_first) ;;
    *) echo "Unknown decoding strategy: $STRATEGY" >&2; exit 2 ;;
esac
RERANK_ARGS=()
STEPS=${STEPS:-10}
if [[ "${SELF_RERANK:-0}" == 1 ]]; then
    RERANK_ARGS+=(--self-rerank)
    STEPS=${REFINEMENT_STEPS:-8}
fi
mkdir -p "$OUTPUT"
# --remove-bpe retains Moses tokenization for tokenized BLEU evaluation.
fairseq-generate "$DATA" --user-dir "$ROOT/diformer" --task diformer_translation \
    --path "$CHECKPOINT" --source-lang "$SRC" --target-lang "$TGT" \
    --left-pad-target false --decode-strategy "$STRATEGY" \
    --refinement-steps "$STEPS" --length-beam "${LENGTH_BEAM:-5}" \
    --beam "${AR_BEAM:-4}" --nbest "${NBEST:-1}" \
    --batch-size "${BATCH_SIZE:-32}" --gen-subset "${SUBSET:-test}" \
    --remove-bpe --results-path "$OUTPUT" "${RERANK_ARGS[@]}" "$@"
