#!/usr/bin/env python3
"""
prepare_stt.py – Build STT/ASR training manifests from multiple high-quality datasets.

Datasets pulled from HuggingFace Hub (streaming where possible):
  1. openslr/librispeech_asr           – clean 960h, supervised transcripts
  2. speechcolab/gigaspeech            – 10k-hour curated web speech (xl split)
  3. mozilla-foundation/common_voice_17_0 – crowd-sourced, diverse accents
  4. facebook/voxpopuli                – parliamentary speech, multilingual

Output JSONL schema (one sample per line):
  {
    "task":       "asr",
    "text_input": "",
    "text_target": "<transcript>",
    "audio_path":  "/absolute/path/to/clip.flac"
  }

Usage:
  python scripts/prepare_stt.py --out data/asr_train.jsonl --cache_dir /data/hf_cache
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import soundfile as sf
from pathlib import Path

from datasets import load_dataset, Audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",          default="data/asr_train.jsonl")
    p.add_argument("--cache_dir",    default="/data/hf_cache")
    p.add_argument("--audio_dir",    default="data/audio/asr",
                   help="Directory where decoded audio clips are saved.")
    p.add_argument("--max_per_ds",   type=int, default=200_000,
                   help="Max samples per source dataset (use 0 for all).")
    p.add_argument("--librispeech",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gigaspeech",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--commonvoice",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--voxpopuli",    action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def save_audio(sample: dict, audio_dir: Path, uid: str) -> str | None:
    """Save a HF audio sample to disk; return the file path or None on error."""
    try:
        arr   = sample["array"]
        sr    = sample["sampling_rate"]
        fpath = audio_dir / f"{uid}.flac"
        sf.write(str(fpath), arr, sr)
        return str(fpath)
    except Exception as exc:
        log.warning("Could not save audio %s: %s", uid, exc)
        return None


def write_rows(rows: list[dict], fh) -> None:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = args.max_per_ds or None

    with out_path.open("w", encoding="utf-8") as fh:

        # ── 1. LibriSpeech clean 360h + 500h ──────────────────────────────────
        if args.librispeech:
            log.info("Loading librispeech_asr …")
            for split_name in ("train.clean.360", "train.other.500"):
                ds = load_dataset(
                    "openslr/librispeech_asr", split=split_name,
                    cache_dir=args.cache_dir, streaming=False,
                ).cast_column("audio", Audio(sampling_rate=16000))
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                rows = []
                for i, ex in enumerate(ds):
                    uid  = f"ls_{split_name.replace('.','_')}_{i:07d}"
                    path = save_audio(ex["audio"], audio_dir, uid)
                    if path:
                        rows.append({
                            "task":        "asr",
                            "text_input":  "",
                            "text_target": ex["text"].strip(),
                            "audio_path":  path,
                        })
                write_rows(rows, fh)
                log.info("LibriSpeech %s: %d rows", split_name, len(rows))

        # ── 2. GigaSpeech XL ──────────────────────────────────────────────────
        if args.gigaspeech:
            log.info("Loading GigaSpeech xl …")
            ds = load_dataset(
                "speechcolab/gigaspeech", "xl", split="train",
                cache_dir=args.cache_dir, trust_remote_code=True,
            ).cast_column("audio", Audio(sampling_rate=16000))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"giga_{i:07d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                if path:
                    rows.append({
                        "task":        "asr",
                        "text_input":  "",
                        "text_target": ex["text"].strip(),
                        "audio_path":  path,
                    })
            write_rows(rows, fh)
            log.info("GigaSpeech: %d rows", len(rows))

        # ── 3. CommonVoice 17.0 (English) ─────────────────────────────────────
        if args.commonvoice:
            log.info("Loading CommonVoice 17.0 en …")
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0", "en", split="train",
                cache_dir=args.cache_dir, trust_remote_code=True,
            ).cast_column("audio", Audio(sampling_rate=16000))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"cv17_{i:07d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                if path:
                    rows.append({
                        "task":        "asr",
                        "text_input":  "",
                        "text_target": ex["sentence"].strip(),
                        "audio_path":  path,
                    })
            write_rows(rows, fh)
            log.info("CommonVoice 17: %d rows", len(rows))

        # ── 4. VoxPopuli (English) ─────────────────────────────────────────────
        if args.voxpopuli:
            log.info("Loading VoxPopuli en …")
            ds = load_dataset(
                "facebook/voxpopuli", "en", split="train",
                cache_dir=args.cache_dir, trust_remote_code=True,
            ).cast_column("audio", Audio(sampling_rate=16000))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            rows = []
            for i, ex in enumerate(ds):
                uid  = f"vox_{i:07d}"
                path = save_audio(ex["audio"], audio_dir, uid)
                if path and ex.get("normalized_text", "").strip():
                    rows.append({
                        "task":        "asr",
                        "text_input":  "",
                        "text_target": ex["normalized_text"].strip(),
                        "audio_path":  path,
                    })
            write_rows(rows, fh)
            log.info("VoxPopuli: %d rows", len(rows))

    log.info("All ASR done → %s", args.out)


if __name__ == "__main__":
    main()
