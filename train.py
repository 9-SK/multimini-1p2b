from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

from multimini.data import (
    MultiTaskCollator,
    MultiTaskManifestDataset,
    WeightedMultiManifestDataset,
)
from multimini.modeling import MultiMiniConfig, MultiMiniModel, count_trainable_parameters
from multimini.tokenizer_utils import build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MultiMini 1.2B – H100/BF16 multimodal + reasoning trainer"
    )
    p.add_argument("--config",           default="configs/model_1p2b.yaml")
    p.add_argument("--manifest",         default=None,
                   help="Single JSONL manifest; overrides manifests list in config.")
    p.add_argument("--tokenizer",        default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--deepspeed_config", default="configs/deepspeed_zero3_bf16.json")
    p.add_argument("--compile",          action="store_true",
                   help="torch.compile with reduce-overhead (recommended on H100)")
    p.add_argument("--no_flash_attn",    action="store_true",
                   help="Disable Flash-Attention 2 (falls back to SDPA)")
    p.add_argument("--no_augment_audio", action="store_true",
                   help="Disable online audio augmentation")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model_cfg(c: dict, use_flash_attn: bool = True) -> MultiMiniConfig:
    m, mo = c["model"], c["modal"]
    return MultiMiniConfig(
        vocab_size              = int(m["vocab_size"]),
        hidden_size             = int(m["hidden_size"]),
        intermediate_size       = int(m["intermediate_size"]),
        num_hidden_layers       = int(m["num_hidden_layers"]),
        num_attention_heads     = int(m["num_attention_heads"]),
        num_key_value_heads     = int(m.get("num_key_value_heads", m["num_attention_heads"])),
        max_position_embeddings = int(m["max_position_embeddings"]),
        rope_theta              = float(m["rope_theta"]),
        rms_norm_eps            = float(m["rms_norm_eps"]),
        tie_word_embeddings     = bool(m["tie_word_embeddings"]),
        gradient_checkpointing  = bool(m.get("gradient_checkpointing", True)),
        use_flash_attention     = use_flash_attn and bool(m.get("use_flash_attention", True)),
        vision_backbone         = str(mo["vision_backbone"]),
        audio_backbone          = str(mo["audio_backbone"]),
        modal_hidden_size       = int(mo["modal_hidden_size"]),
        projector_hidden_size   = int(mo["projector_hidden_size"]),
        projector_depth         = int(mo.get("projector_depth", 3)),
        freeze_vision_backbone  = bool(mo["freeze_vision_backbone"]),
        freeze_audio_backbone   = bool(mo["freeze_audio_backbone"]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def cosine_with_warmup(
    optimizer: AdamW, warmup: int, total: int, min_ratio: float = 0.1
) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, fn)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_cfg(args.config)
    tcfg = cfg["training"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    # H100-specific tuning
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS",   "1")
    os.environ.setdefault("NCCL_MIN_NCHANNELS",            "4")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS",          "4")
    os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")

    precision  = str(tcfg.get("precision", "bf16"))
    grad_accum = int(tcfg.get("grad_accum_steps", 8))
    accelerator = Accelerator(
        mixed_precision=precision, gradient_accumulation_steps=grad_accum
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError(f"CUDA required, got: {accelerator.device.type}")

    torch.manual_seed(int(tcfg.get("seed", 2026)) + accelerator.process_index)

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    special = list(cfg["special_tokens"].values())
    tokenizer, _ = build_tokenizer(args.tokenizer, special, num_audio_tokens=2048)

    # ── Model ─────────────────────────────────────────────────────────────────
    use_flash = not args.no_flash_attn
    mcfg  = build_model_cfg(cfg, use_flash_attn=use_flash)
    model = MultiMiniModel(mcfg)
    model.decoder.resize_token_embeddings(len(tokenizer))

    if args.compile:
        logger.info("torch.compile(reduce-overhead) …")
        model = torch.compile(model, mode="reduce-overhead")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    # Resolve vision processor (works whether compiled or not)
    raw_model = getattr(model, "_orig_mod", model)
    vproc  = raw_model.vision_processor
    aproc  = AutoProcessor.from_pretrained(mcfg.audio_backbone)
    max_tl = int(tcfg.get("max_text_len", 4096))
    collator = MultiTaskCollator(tokenizer, vproc, aproc, max_text_len=max_tl)

    augment = not args.no_augment_audio
    if args.manifest:
        dataset = MultiTaskManifestDataset(args.manifest, augment_audio=augment)
        sampler = None
    else:
        manifests    = cfg.get("manifests", [])
        task_weights = cfg.get("task_weights", {})
        if not manifests:
            raise ValueError("Pass --manifest or add a 'manifests' list to the config.")
        weights = [
            float(task_weights.get(Path(m).stem.rsplit("_", 1)[0], 1.0))
            for m in manifests
        ]
        dataset = WeightedMultiManifestDataset(manifests, weights=weights, augment_audio=augment)
        sampler = dataset.make_sampler()

    num_workers = int(tcfg.get("num_workers", 8))
    dl_kwargs   = {"prefetch_factor": int(tcfg.get("prefetch_factor", 4))} if num_workers > 0 else {}
    dataloader = DataLoader(
        dataset,
        batch_size         = int(tcfg.get("batch_size_per_gpu", 8)),
        sampler            = sampler,
        shuffle            = sampler is None,
        drop_last          = True,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = num_workers > 0,
        collate_fn         = collator,
        **dl_kwargs,
    )

    # ── Optimiser ──────────────────────────────────────────────────────────────
    lr  = float(tcfg.get("learning_rate", 1.5e-4))
    optimizer = AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = float(tcfg.get("weight_decay", 0.1)),
        betas        = tuple(tcfg.get("betas", [0.9, 0.95])),
        eps          = float(tcfg.get("eps",   1e-8)),
        fused        = True,
    )

    max_steps    = int(tcfg.get("max_steps",    500000))
    warmup_steps = int(tcfg.get("warmup_steps", 2000))
    min_lr_ratio = float(tcfg.get("min_lr", lr * 0.1)) / lr
    scheduler    = cosine_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    save_every    = int(tcfg.get("save_every",    2000))
    log_every     = int(tcfg.get("log_every",     10))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    out_dir       = Path(str(tcfg.get("output_dir", "checkpoints/multimini-1p2b")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    resume_from = str(tcfg.get("resume_from", "")).strip()
    if resume_from and Path(resume_from).exists():
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(state["model"], strict=False)
        if "optimizer"  in state: optimizer.load_state_dict(state["optimizer"])
        if "scheduler"  in state: scheduler.load_state_dict(state["scheduler"])
        global_step = int(state.get("global_step", 0))
        logger.info("Resumed from step %d", global_step)

    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        logger.info(json.dumps({
            "trainable_params":    count_trainable_parameters(raw),
            "tokenizer_size":      len(tokenizer),
            "precision":           precision,
            "flash_attention":     use_flash,
            "compile":             args.compile,
            "eff_batch":           int(tcfg.get("batch_size_per_gpu", 8))
                                   * grad_accum * accelerator.num_processes,
        }, indent=2))

    # ── Training loop ──────────────────────────────────────────────────────────
    progress     = tqdm(total=max_steps, initial=global_step,
                        disable=not accelerator.is_main_process, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    def _save(step: int) -> None:
        ckpt = {
            "model":       accelerator.unwrap_model(model).state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "global_step": step,
        }
        p = out_dir / f"step-{step}.pt"
        accelerator.save(ckpt, p)
        if accelerator.is_main_process:
            logger.info("Checkpoint → %s", p)

    while global_step < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                out  = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = batch["labels"],
                    image_pixels   = batch["image_pixels"],
                    audio_features = batch["audio_features"],
                )
                loss = out.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)

                    running_loss += accelerator.gather(loss.detach()).mean().item()

                    if accelerator.is_main_process and global_step % log_every == 0:
                        avg_loss   = running_loss / log_every
                        running_loss = 0.0
                        cur_lr     = scheduler.get_last_lr()[0]
                        progress.set_description(
                            f"step={global_step}  loss={avg_loss:.4f}  lr={cur_lr:.2e}"
                        )

                    if global_step % save_every == 0:
                        _save(global_step)

                    if global_step >= max_steps:
                        break

        if global_step >= max_steps:
            break

    _save(global_step)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training complete at step %d", global_step)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
