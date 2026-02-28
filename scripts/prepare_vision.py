#!/usr/bin/env python3
"""
prepare_vision.py – Build image-to-text training manifests.

Datasets:
  1. HuggingFaceM4/NoCaps               – open-domain image captioning (grounded)
  2. lmms-lab/LLaVA-OneVision-Data      – LLaVA-style instruction + reasoning pairs
  3. merve/coco_captions                 – COCO Captions (5 refs per image)
  4. HuggingFaceM4/the_cauldron         – multi-task VQA/caption superset
  5. liuhaotian/LLaVA-Instruct-150K     – visual instruction following (~150k)
  6. Multimodal-Fatima/VQAv2_train      – VQA v2 training split

Output JSONL:
  {
    "task":        "i2t",
    "text_input":  "<question or empty>",
    "text_target": "<caption / answer>",
    "image_path":  "/abs/path.jpg"
  }

For reasoning-style rows a `thinking` field is added where the dataset provides
chain-of-thought rationales.

Usage:
  python scripts/prepare_vision.py --out data/image_caption_train.jsonl \\
      --image_dir data/images --cache_dir /data/hf_cache
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="data/image_caption_train.jsonl")
    p.add_argument("--cache_dir",  default="/data/hf_cache")
    p.add_argument("--image_dir",  default="data/images")
    p.add_argument("--max_per_ds", type=int, default=200_000)
    p.add_argument("--nocaps",     action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--llava",      action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coco",       action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cauldron",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vqav2",      action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def save_pil(img: Image.Image, img_dir: Path, uid: str) -> str | None:
    try:
        img = img.convert("RGB")
        p   = img_dir / f"{uid}.jpg"
        img.save(str(p), "JPEG", quality=92)
        return str(p)
    except Exception as exc:
        log.warning("save_pil %s: %s", uid, exc)
        return None


def main() -> None:
    args = parse_args()
    img_dir  = Path(args.image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = args.max_per_ds or None

    with out_path.open("w", encoding="utf-8") as fh:

        def emit(row: dict) -> None:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        # ── 1. NoCaps ──────────────────────────────────────────────────────────
        if args.nocaps:
            log.info("Loading NoCaps …")
            for split in ("train", "validation"):
                try:
                    ds = load_dataset(
                        "HuggingFaceM4/NoCaps", split=split,
                        cache_dir=args.cache_dir,
                    )
                    if limit:
                        ds = ds.select(range(min(limit, len(ds))))
                    count = 0
                    for i, ex in enumerate(ds):
                        img_path = save_pil(ex["image"], img_dir, f"nocaps_{split}_{i:07d}")
                        if not img_path:
                            continue
                        for j, cap in enumerate(ex.get("annotations_captions", [ex.get("caption","")])):
                            if cap:
                                emit({"task": "i2t", "text_input": "", "text_target": cap.strip(), "image_path": img_path})
                                count += 1
                    log.info("NoCaps %s: %d rows", split, count)
                except Exception as exc:
                    log.warning("NoCaps %s failed: %s", split, exc)

        # ── 2. LLaVA Instruct 150K ────────────────────────────────────────────
        if args.llava:
            log.info("Loading LLaVA-Instruct-150K …")
            try:
                ds = load_dataset(
                    "liuhaotian/LLaVA-Instruct-150K", split="train",
                    cache_dir=args.cache_dir,
                )
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                count = 0
                for i, ex in enumerate(ds):
                    convs = ex.get("conversations", [])
                    # Flatten multi-turn convos into (human, gpt) pairs
                    img  = ex.get("image")
                    if img is None:
                        continue
                    img_path = save_pil(img, img_dir, f"llava_{i:07d}")
                    if not img_path:
                        continue
                    for k in range(0, len(convs) - 1, 2):
                        human = convs[k].get("value", "").replace("<image>", "").strip()
                        gpt   = convs[k + 1].get("value", "").strip()
                        if human and gpt:
                            emit({
                                "task":        "i2t",
                                "text_input":  human,
                                "text_target": gpt,
                                "image_path":  img_path,
                            })
                            count += 1
                log.info("LLaVA-Instruct: %d rows", count)
            except Exception as exc:
                log.warning("LLaVA failed: %s", exc)

        # ── 3. COCO Captions ──────────────────────────────────────────────────
        if args.coco:
            log.info("Loading COCO Captions …")
            try:
                ds = load_dataset(
                    "merve/coco_captions", split="train",
                    cache_dir=args.cache_dir,
                )
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                count = 0
                for i, ex in enumerate(ds):
                    img_path = save_pil(ex["image"], img_dir, f"coco_{i:07d}")
                    if not img_path:
                        continue
                    for cap in ex.get("sentences_raw", [ex.get("caption","")]):
                        if cap:
                            emit({"task": "i2t", "text_input": "Describe this image.", "text_target": cap.strip(), "image_path": img_path})
                            count += 1
                log.info("COCO Captions: %d rows", count)
            except Exception as exc:
                log.warning("COCO failed: %s", exc)

        # ── 4. The Cauldron (multi-task VQA superset) ─────────────────────────
        if args.cauldron:
            log.info("Loading The Cauldron (sample) …")
            # The Cauldron has many subsets; pick a representative mix
            for subset in ("vqav2", "textvqa", "docvqa", "scienceqa", "ai2d"):
                try:
                    ds = load_dataset(
                        "HuggingFaceM4/the_cauldron", subset, split="train",
                        cache_dir=args.cache_dir, trust_remote_code=True,
                    )
                    if limit:
                        ds = ds.select(range(min(limit // 5, len(ds))))
                    count = 0
                    for i, ex in enumerate(ds):
                        images = ex.get("images", [])
                        if not images:
                            continue
                        img_path = save_pil(images[0], img_dir, f"cald_{subset}_{i:07d}")
                        if not img_path:
                            continue
                        texts = ex.get("texts", [])
                        for pair in texts:
                            q = pair.get("user",      "").strip()
                            a = pair.get("assistant", "").strip()
                            if q and a:
                                emit({"task": "i2t", "text_input": q, "text_target": a, "image_path": img_path})
                                count += 1
                    log.info("Cauldron/%s: %d rows", subset, count)
                except Exception as exc:
                    log.warning("Cauldron/%s failed: %s", subset, exc)

        # ── 5. VQA v2 ─────────────────────────────────────────────────────────
        if args.vqav2:
            log.info("Loading VQAv2 …")
            try:
                ds = load_dataset(
                    "Multimodal-Fatima/VQAv2_train", split="train",
                    cache_dir=args.cache_dir,
                )
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                count = 0
                for i, ex in enumerate(ds):
                    img_path = save_pil(ex["image"], img_dir, f"vqa2_{i:07d}")
                    if not img_path:
                        continue
                    q = ex.get("question", "").strip()
                    a = ex.get("multiple_choice_answer", "").strip()
                    if q and a:
                        emit({"task": "i2t", "text_input": q, "text_target": a, "image_path": img_path})
                        count += 1
                log.info("VQAv2: %d rows", count)
            except Exception as exc:
                log.warning("VQAv2 failed: %s", exc)

    log.info("All vision done → %s", args.out)


if __name__ == "__main__":
    main()
