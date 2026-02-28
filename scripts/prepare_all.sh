#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# prepare_all.sh – Run every dataset preparation script in sequence.
#
# Usage:
#   bash scripts/prepare_all.sh [--cache_dir /data/hf_cache] [--max_per_ds 200000]
#
# Creates:
#   data/chat_reasoning.jsonl
#   data/code_reasoning.jsonl
#   data/math_reasoning.jsonl
#   data/asr_train.jsonl
#   data/tts_train.jsonl
#   data/image_caption_train.jsonl
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CACHE_DIR="${CACHE_DIR:-/data/hf_cache}"
MAX="${MAX_PER_DS:-200000}"

PYTHON="${PYTHON:-python}"

echo "==> Preparing chat + reasoning data …"
$PYTHON scripts/prepare_chat_reasoning.py \
    --out data/chat_reasoning.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo "==> Preparing code + reasoning data …"
$PYTHON scripts/prepare_code.py \
    --out data/code_reasoning.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo "==> Preparing math + reasoning data …"
$PYTHON scripts/prepare_math.py \
    --out data/math_reasoning.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo "==> Preparing STT / ASR data …"
$PYTHON scripts/prepare_stt.py \
    --out data/asr_train.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo "==> Preparing TTS data …"
$PYTHON scripts/prepare_tts.py \
    --out data/tts_train.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo "==> Preparing vision (image-to-text) data …"
$PYTHON scripts/prepare_vision.py \
    --out data/image_caption_train.jsonl \
    --cache_dir "$CACHE_DIR" --max_per_ds "$MAX"

echo ""
echo "All datasets ready. Row counts:"
for f in data/*.jsonl; do
    count=$(wc -l < "$f")
    printf "  %-40s %d rows\n" "$f" "$count"
done
