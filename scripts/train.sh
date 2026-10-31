#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 5 )); then
    echo "Usage: bash scripts/train.sh {wmt14_en_de|wmt16_en_ro} DATA_BIN SOURCE TARGET SAVE_DIR [fairseq-train options]" >&2
    exit 2
fi
PRESET=$1
DATA=$2
SRC=$3
TGT=$4
SAVE=$5
shift 5
case "$PRESET" in
    wmt14_en_de|wmt16_en_ro) source "$ROOT/configs/$PRESET.sh" ;;
    *) echo "Unknown preset: $PRESET" >&2; exit 2 ;;
esac
GPU_COUNT=${GPUS:-8}
TOKENS=${MAX_TOKENS:-8192}
ACCUMULATION=${UPDATE_FREQ:-2}
# Use --fp16 as an extra option on supported GPUs; never force it on CPU.
fairseq-train "$DATA" --user-dir "$ROOT/diformer" \
    --task diformer_translation --arch diformer --criterion nat_loss --noise no_noise \
    --source-lang "$SRC" --target-lang "$TGT" --save-dir "$SAVE" \
    --left-pad-target false --share-all-embeddings \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --encoder-learned-pos --decoder-learned-pos --apply-bert-init \
    --rel-pos-max-distance 256 --length-loss-factor 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-8 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --warmup-init-lr 1e-7 --label-smoothing 0.1 --dropout "$DROPOUT" \
    --weight-decay 0.01 --clip-norm 0.0 \
    --distributed-world-size "$GPU_COUNT" --ddp-backend legacy_ddp \
    --max-tokens "$TOKENS" --update-freq "$ACCUMULATION" \
    --max-update "${TRAIN_UPDATES:-$MAX_UPDATES}" --seed "${SEED:-1}" \
    --save-interval-updates 1000 --keep-interval-updates 5 --no-epoch-checkpoints \
    --log-interval 100 "$@"
