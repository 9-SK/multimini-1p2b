#!/usr/bin/env python3
"""
prepare_code.py – High-quality coding + code-reasoning manifests.

Datasets:
  1. open-thoughts/OpenThoughts-114k  – 114k code+math reasoning traces (DeepSeek-R1 distilled)
  2. m-a-p/OpenCodeInterpreter-OS     – code execution + explanation pairs
  3. ise-uiuc/Magicoder-OSS-Instruct-75K – OS-code instruction pairs
  4. bigcode/self-oss-instruct-sc2-exec-filter-50k  – self-instruct, exec-verified
  5. HuggingFaceH4/CodeAlpaca_20K     – 20k coding instructions
  6. nampdn-ai/tiny-codes             – compact diverse coding examples

Schema:
  {
    "task":        "code",
    "text_input":  "<problem / instruction>",
    "text_target": "<solution>",
    "thinking":    "<reasoning trace if available>"
  }

Usage:
  python scripts/prepare_code.py --out data/code_reasoning.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="data/code_reasoning.jsonl")
    p.add_argument("--cache_dir",  default="/data/hf_cache")
    p.add_argument("--max_per_ds", type=int, default=200_000)
    return p.parse_args()


def emit(fh, text_input: str, text_target: str, thinking: str = "") -> None:
    row: dict = {"task": "code", "text_input": text_input, "text_target": text_target}
    if thinking:
        row["thinking"] = thinking
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args     = parse_args()
    limit    = args.max_per_ds or None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:

        # ── 1. OpenThoughts-114k (R1-distilled code+math reasoning) ──────────
        log.info("OpenThoughts-114k …")
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                # Skip non-code examples (math handled separately)
                if ex.get("subject", "code").lower() not in ("code", "coding", "programming"):
                    continue
                problem  = (ex.get("problem")  or ex.get("instruction") or "").strip()
                thinking = (ex.get("thinking") or ex.get("reasoning")   or "").strip()
                solution = (ex.get("solution") or ex.get("response")    or "").strip()
                if problem and solution:
                    emit(fh, problem, solution, thinking=thinking)
                    count += 1
            log.info("OpenThoughts (code): %d", count)
        except Exception as e:
            log.warning("OpenThoughts failed: %s", e)

        # ── 2. OpenCodeInterpreter ────────────────────────────────────────────
        log.info("OpenCodeInterpreter …")
        try:
            ds = load_dataset("m-a-p/OpenCodeInterpreter-OS", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                convs = ex.get("conversations", [])
                for k in range(0, len(convs) - 1, 2):
                    u = convs[k].get("value", "").strip()
                    a = convs[k + 1].get("value", "").strip()
                    if u and a:
                        emit(fh, u, a)
                        count += 1
            log.info("OpenCodeInterpreter: %d", count)
        except Exception as e:
            log.warning("OpenCodeInterpreter failed: %s", e)

        # ── 3. Magicoder OSS-Instruct 75k ─────────────────────────────────────
        log.info("Magicoder-OSS-Instruct-75K …")
        try:
            ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = ex.get("problem", ex.get("instruction", "")).strip()
                a = ex.get("solution", ex.get("response", "")).strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("Magicoder: %d", count)
        except Exception as e:
            log.warning("Magicoder failed: %s", e)

        # ── 4. bigcode self-oss-instruct (exec-verified) ──────────────────────
        log.info("self-oss-instruct-sc2-exec-filter-50k …")
        try:
            ds = load_dataset(
                "bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train",
                cache_dir=args.cache_dir,
            )
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = ex.get("instruction", ex.get("prompt", "")).strip()
                a = ex.get("response",    ex.get("completion", "")).strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("self-oss-instruct: %d", count)
        except Exception as e:
            log.warning("self-oss-instruct failed: %s", e)

        # ── 5. CodeAlpaca 20k ─────────────────────────────────────────────────
        log.info("CodeAlpaca 20k …")
        try:
            ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = (ex.get("prompt") or ex.get("instruction") or "").strip()
                a = (ex.get("completion") or ex.get("output") or "").strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("CodeAlpaca: %d", count)
        except Exception as e:
            log.warning("CodeAlpaca failed: %s", e)

        # ── 6. tiny-codes ─────────────────────────────────────────────────────
        log.info("tiny-codes …")
        try:
            ds = load_dataset("nampdn-ai/tiny-codes", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = ex.get("prompt", "").strip()
                a = ex.get("response", "").strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("tiny-codes: %d", count)
        except Exception as e:
            log.warning("tiny-codes failed: %s", e)

    log.info("Code done → %s", args.out)


if __name__ == "__main__":
    main()
