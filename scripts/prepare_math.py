#!/usr/bin/env python3
"""
prepare_math.py – High-quality math + mathematical reasoning manifests.

Datasets:
  1. open-thoughts/OpenThoughts-114k   – R1-distilled math reasoning traces
  2. AI-MO/NuminaMath-CoT             – 860k competition math + CoT
  3. lighteval/MATH                    – 12.5k AMC/AIME/competition problems
  4. microsoft/orca-math-word-problems-200k – 200k GPT-4 word problems
  5. HuggingFaceH4/MATH-500           – hard math eval (use for data diversity)
  6. qwedsacf/grade-school-math        – GSM8K: grade-school multi-step

Schema:
  {
    "task":        "math",
    "text_input":  "<problem>",
    "text_target": "<final answer>",
    "thinking":    "<step-by-step reasoning>"
  }

Usage:
  python scripts/prepare_math.py --out data/math_reasoning.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from datasets import load_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Regex to extract boxed final answer from LaTeX \boxed{...}
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_HASH  = re.compile(r"####\s*(.+)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="data/math_reasoning.jsonl")
    p.add_argument("--cache_dir",  default="/data/hf_cache")
    p.add_argument("--max_per_ds", type=int, default=300_000)
    return p.parse_args()


def extract_answer(text: str) -> tuple[str, str]:
    """Return (thinking, final_answer) extracted from a full solution string."""
    # GSM8K style: #### answer
    m = _HASH.search(text)
    if m:
        answer   = m.group(1).strip()
        thinking = text[: m.start()].strip()
        return thinking, answer
    # LaTeX boxed
    m = _BOXED.search(text)
    if m:
        answer   = m.group(1).strip()
        thinking = text
        return thinking, answer
    # Fallback: entire text is thinking, no separate answer
    return text, ""


def emit(fh, problem: str, thinking: str, answer: str) -> None:
    if not (problem and answer):
        return
    row: dict = {"task": "math", "text_input": problem, "text_target": answer}
    if thinking:
        row["thinking"] = thinking
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args     = parse_args()
    limit    = args.max_per_ds or None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:

        # ── 1. OpenThoughts-114k (math slice) ─────────────────────────────────
        log.info("OpenThoughts-114k (math) …")
        try:
            ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                subj = ex.get("subject", "").lower()
                if subj not in ("math", "mathematics", ""):
                    continue
                problem  = (ex.get("problem")  or ex.get("instruction") or "").strip()
                thinking = (ex.get("thinking") or ex.get("reasoning")   or "").strip()
                solution = (ex.get("solution") or ex.get("response")    or "").strip()
                _, answer = extract_answer(solution)
                if not answer:
                    answer = solution
                if problem:
                    emit(fh, problem, thinking, answer)
                    count += 1
            log.info("OpenThoughts (math): %d", count)
        except Exception as e:
            log.warning("OpenThoughts failed: %s", e)

        # ── 2. NuminaMath-CoT ─────────────────────────────────────────────────
        log.info("NuminaMath-CoT …")
        try:
            ds = load_dataset("AI-MO/NuminaMath-CoT", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                problem  = (ex.get("problem") or "").strip()
                solution = (ex.get("solution") or "").strip()
                thinking, answer = extract_answer(solution)
                if not answer:
                    answer = solution
                if problem:
                    emit(fh, problem, thinking, answer)
                    count += 1
            log.info("NuminaMath-CoT: %d", count)
        except Exception as e:
            log.warning("NuminaMath failed: %s", e)

        # ── 3. MATH (competition) ─────────────────────────────────────────────
        log.info("MATH …")
        try:
            ds = load_dataset("lighteval/MATH", "all", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                problem  = (ex.get("problem") or "").strip()
                solution = (ex.get("solution") or "").strip()
                thinking, answer = extract_answer(solution)
                if not answer:
                    answer = solution
                if problem:
                    emit(fh, problem, thinking, answer)
                    count += 1
            log.info("MATH: %d", count)
        except Exception as e:
            log.warning("MATH failed: %s", e)

        # ── 4. Orca Math Word Problems 200k ───────────────────────────────────
        log.info("Orca Math 200k …")
        try:
            ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                problem  = (ex.get("question") or "").strip()
                solution = (ex.get("answer")   or "").strip()
                thinking, answer = extract_answer(solution)
                if not answer:
                    answer = solution
                if problem:
                    emit(fh, problem, thinking, answer)
                    count += 1
            log.info("Orca Math: %d", count)
        except Exception as e:
            log.warning("Orca Math failed: %s", e)

        # ── 5. GSM8K (grade school math multi-step) ───────────────────────────
        log.info("GSM8K …")
        try:
            ds = load_dataset("qwedsacf/grade-school-math", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                problem  = (ex.get("question") or "").strip()
                solution = (ex.get("answer")   or "").strip()
                thinking, answer = extract_answer(solution)
                if not answer:
                    answer = solution
                if problem:
                    emit(fh, problem, thinking, answer)
                    count += 1
            log.info("GSM8K: %d", count)
        except Exception as e:
            log.warning("GSM8K failed: %s", e)

    log.info("Math done → %s", args.out)


if __name__ == "__main__":
    main()
