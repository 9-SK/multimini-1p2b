#!/usr/bin/env python3
"""
prepare_tts.py – Build TTS training manifests from high-quality speech corpora.

Datasets:
  1. keithito/lj_speech      – LJ Speech 24 h single-speaker studio quality
  2. facebook/multilingual_librispeech (en)  – MLS English 44.5 h
  3. VCTK (via openslr/cmu-arctic or Hubert/vctk-corpus-speech-dataset)
  4. blabble-io/libritts      – LibriTTS-R: studio-cleaned LibriSpeech
  5. WillHeld/vctk            – VCTK multi-speaker

Output JSONL:
  {
    "task":        "tts",
    "text_input":  "<transcript>",
    "text_target": "",
    "audio_path":  "/abs/path.flac"   <- audio TARGET for codec supervision
  }

Usage:
  python scripts/prepare_tts.py --out data/tts_train.jsonl --cache_dir /data/hf_cache
"""
from __future__ import annotations

import argparse
import json
import logging
import soundfile as sf
from pathlib import Path

from datasets import load_dataset, Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="data/tts_train.jsonl")
    p.add_argument("--cache_dir",  default="/data/hf_cache")
    p.add_argument("--audio_dir",  default="data/audio/tts")
    p.add_argument("--max_per_ds", type=int, default=200_000)
    p.add_argument("--ljspeech",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mls",        action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--libritts",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vctk",       action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def save_audio(sample: dict, audio_dir: Path, uid: str) -> str | None:
    try:
        fpath = audio_dir / f"{uid}.flac"
        sf.write(str(fpath), sample["array"], sample["sampling_rate"])
        return str(fpath)
    except Exception as exc:
        log.warning("save_audio %s: %s", uid, exc)
        return None


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = args.max_per_ds or None

    with out_path.open("w", encoding="utf-8") as fh:

        # ── 1. LJ Speech ──────────────────────────────────────────────────────
        if args.ljspeech:
            log.info("Loading LJ Speech …")
            ds = load_dataset(
                "keithito/lj_speech", split="train",
                cache_dir=args.cache_dir,
            ).cast_column("audio", Audio(sampling_rate=22050))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"lj_{i:06d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                if path:
                    rows.append({
                        "task":        "tts",
                        "text_input":  ex["normalized_text"].strip(),
                        "text_target": "",
                        "audio_path":  path,
                    })
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info("LJ Speech: %d rows", len(rows))

        # ── 2. MLS English ────────────────────────────────────────────────────
        if args.mls:
            log.info("Loading MLS English …")
            ds = load_dataset(
                "facebook/multilingual_librispeech", "english", split="train",
                cache_dir=args.cache_dir,
            ).cast_column("audio", Audio(sampling_rate=16000))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"mls_{i:07d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                if path and ex.get("text", "").strip():
                    rows.append({
                        "task":        "tts",
                        "text_input":  ex["text"].strip(),
                        "text_target": "",
                        "audio_path":  path,
                    })
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info("MLS English: %d rows", len(rows))

        # ── 3. LibriTTS-R (clean) ─────────────────────────────────────────────
        if args.libritts:
            log.info("Loading LibriTTS-R clean …")
            for split_name in ("train.clean.360", "train.clean.100"):
                ds = load_dataset(
                    "blabble-io/libritts", split=split_name,
                    cache_dir=args.cache_dir, trust_remote_code=True,
                ).cast_column("audio", Audio(sampling_rate=24000))
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                rows = []
                for i, ex in enumerate(ds):
                    uid  = f"ltts_{split_name.replace('.','_')}_{i:07d}"
                    path = save_audio(ex["audio"], audio_dir, uid)
                    transcript = (ex.get("text_normalized") or ex.get("text", "")).strip()
                    if path and transcript:
                        rows.append({
                            "task":        "tts",
                            "text_input":  transcript,
                            "text_target": "",
                            "audio_path":  path,
                        })
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                log.info("LibriTTS-R %s: %d rows", split_name, len(rows))

        # ── 4. VCTK multi-speaker ─────────────────────────────────────────────
        if args.vctk:
            log.info("Loading VCTK …")
            ds = load_dataset(
                "WillHeld/vctk", split="train",
                cache_dir=args.cache_dir, trust_remote_code=True,
            ).cast_column("audio", Audio(sampling_rate=22050))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"vctk_{i:06d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                transcript = (ex.get("text") or "").strip()
                if path and transcript:
                    rows.append({
                        "task":        "tts",
                        "text_input":  transcript,
                        "text_target": "",
                        "audio_path":  path,
                    })
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info("VCTK: %d rows", len(rows))

    log.info("All TTS done → %s", args.out)


if __name__ == "__main__":
    main()
