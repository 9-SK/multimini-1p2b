#!/usr/bin/env python3
"""
prepare_chat_reasoning.py – High-quality chat + chain-of-thought reasoning manifests.

Datasets:
  1. teknium/OpenHermes-2.5          – 1M curated instruction pairs (GPT-4 sourced)
  2. Open-Orca/SlimOrca              – 518k deduped Orca reasoning traces
  3. HuggingFaceH4/ultrachat_200k    – 200k multi-turn human-like conversations
  4. allenai/tulu-3-sft-mixture      – TULU-3 diverse SFT superset
  5. nvidia/HelpSteer2               – helpfulness-ranked preference pairs
  6. argilla/magpie-ultra-v0.1       – ultra-quality Magpie self-play pairs

Schema:
  {
    "task":        "chat",
    "text_input":  "<user turn>",
    "text_target": "<assistant turn>",
    "thinking":    "<cot trace if available>"   # auto-wrapped in <|think|>
  }

Usage:
  python scripts/prepare_chat_reasoning.py --out data/chat_reasoning.jsonl
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
    p.add_argument("--out",        default="data/chat_reasoning.jsonl")
    p.add_argument("--cache_dir",  default="/data/hf_cache")
    p.add_argument("--max_per_ds", type=int, default=300_000)
    return p.parse_args()


def emit(fh, text_input: str, text_target: str, thinking: str = "") -> None:
    row: dict = {"task": "chat", "text_input": text_input, "text_target": text_target}
    if thinking:
        row["thinking"] = thinking
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args    = parse_args()
    limit   = args.max_per_ds or None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:

        # ── 1. OpenHermes 2.5 ─────────────────────────────────────────────────
        log.info("OpenHermes-2.5 …")
        try:
            ds = load_dataset("teknium/OpenHermes-2.5", split="train",
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
            log.info("OpenHermes-2.5: %d", count)
        except Exception as e:
            log.warning("OpenHermes failed: %s", e)

        # ── 2. SlimOrca (with reasoning traces) ───────────────────────────────
        log.info("SlimOrca …")
        try:
            ds = load_dataset("Open-Orca/SlimOrca", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                convs = ex.get("conversations", [])
                system = ""
                pairs: list[tuple[str,str]] = []
                for turn in convs:
                    role = turn.get("from", turn.get("role", ""))
                    val  = turn.get("value", turn.get("content", "")).strip()
                    if role == "system":
                        system = val
                    elif role in ("human", "user"):
                        pairs.append((val, ""))
                    elif role in ("gpt", "assistant") and pairs:
                        pairs[-1] = (pairs[-1][0], val)
                for u, a in pairs:
                    if u and a:
                        # SlimOrca includes step-by-step reasoning in the answer;
                        # split on "####" if present (GSM8K style)
                        if "####" in a:
                            thinking_part, answer_part = a.split("####", 1)
                            emit(fh, u, answer_part.strip(), thinking=thinking_part.strip())
                        else:
                            emit(fh, u, a)
                        count += 1
            log.info("SlimOrca: %d", count)
        except Exception as e:
            log.warning("SlimOrca failed: %s", e)

        # ── 3. UltraChat 200k ─────────────────────────────────────────────────
        log.info("UltraChat 200k …")
        try:
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                msgs = ex.get("messages", [])
                for k in range(0, len(msgs) - 1, 2):
                    u = msgs[k].get("content", "").strip()
                    a = msgs[k + 1].get("content", "").strip()
                    if u and a:
                        emit(fh, u, a)
                        count += 1
            log.info("UltraChat: %d", count)
        except Exception as e:
            log.warning("UltraChat failed: %s", e)

        # ── 4. TULU-3 SFT mixture ─────────────────────────────────────────────
        log.info("TULU-3 SFT mixture …")
        try:
            ds = load_dataset("allenai/tulu-3-sft-mixture", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                msgs = ex.get("messages", [])
                for k in range(0, len(msgs) - 1, 2):
                    u = msgs[k].get("content", "").strip()
                    a = msgs[k + 1].get("content", "").strip()
                    if u and a:
                        emit(fh, u, a)
                        count += 1
            log.info("TULU-3: %d", count)
        except Exception as e:
            log.warning("TULU-3 failed: %s", e)

        # ── 5. HelpSteer2 ─────────────────────────────────────────────────────
        log.info("HelpSteer2 …")
        try:
            ds = load_dataset("nvidia/HelpSteer2", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = ex.get("prompt", "").strip()
                a = ex.get("response", "").strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("HelpSteer2: %d", count)
        except Exception as e:
            log.warning("HelpSteer2 failed: %s", e)

        # ── 6. Magpie Ultra ───────────────────────────────────────────────────
        log.info("Magpie Ultra …")
        try:
            ds = load_dataset("argilla/magpie-ultra-v0.1", split="train",
                              cache_dir=args.cache_dir)
            if limit: ds = ds.select(range(min(limit, len(ds))))
            count = 0
            for ex in ds:
                u = ex.get("instruction", "").strip()
                a = ex.get("response", "").strip()
                if u and a:
                    emit(fh, u, a)
                    count += 1
            log.info("Magpie Ultra: %d", count)
        except Exception as e:
            log.warning("Magpie Ultra failed: %s", e)

    log.info("Chat/reasoning done → %s", args.out)


if __name__ == "__main__":
    main()
