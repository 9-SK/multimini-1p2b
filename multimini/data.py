from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates per task
# ─────────────────────────────────────────────────────────────────────────────
# Reasoning tasks wrap their target in <|think|>...<|/think|> before the answer.
# The collator handles which tasks get CoT formatting.

REASONING_TASKS = {"chat", "code", "math"}

PROMPT_TEMPLATE = (
    "<|task_{task}|>"
    "{modal_prefix}"
    "<|text|> {text_input}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Audio augmentation
# ─────────────────────────────────────────────────────────────────────────────

class AudioAugmentor:
    """Online augmentations for 16 kHz mono waveforms."""

    def __init__(
        self,
        p_noise: float = 0.3,
        p_speed: float = 0.3,
        noise_std: float = 0.002,
        speed_factors: tuple[float, ...] = (0.9, 0.95, 1.05, 1.1),
    ) -> None:
        self.p_noise = p_noise
        self.p_speed = p_speed
        self.noise_std = noise_std
        self.speed_factors = speed_factors

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_noise:
            wav = (wav + torch.randn_like(wav) * self.noise_std).clamp(-1.0, 1.0)
        if random.random() < self.p_speed:
            factor = random.choice(self.speed_factors)
            orig_len = wav.size(-1)
            new_sr = int(16000 * factor)
            wav = torchaudio.functional.resample(wav.unsqueeze(0), 16000, new_sr).squeeze(0)
            if wav.size(-1) > orig_len:
                wav = wav[:orig_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, orig_len - wav.size(-1)))
        return wav


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskManifestDataset(Dataset):
    """Unified manifest dataset.

    JSONL schema – one sample per line:
    {
      "task":             "chat|code|math|asr|tts|a2a|i2t|i2a",
      "text_input":       "...",
      "text_target":      "...",           # plain answer or <|think|>...<|/think|> answer
      "thinking":         "...",           # optional: raw CoT chain (auto-wrapped)
      "image_path":       "optional",
      "audio_path":       "optional",
      "audio_target_path":"optional"
    }

    When `thinking` is present the collator wraps:
      <|think|>{thinking}<|/think|> {text_target}
    """

    def __init__(self, manifest_path: str, augment_audio: bool = False) -> None:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with path.open("r", encoding="utf-8") as f:
            self.rows: List[Dict] = [json.loads(l) for l in f if l.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {manifest_path}")
        self.augmentor = AudioAugmentor() if augment_audio else None
        logger.info("Loaded %d samples from %s", len(self.rows), manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, path: Optional[str], augment: bool = False) -> Optional[torch.Tensor]:
        if not path:
            return None
        try:
            wav, sr = torchaudio.load(path)
        except Exception as exc:
            logger.warning("Failed to load audio %s: %s", path, exc)
            return None
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        if augment and self.augmentor is not None:
            wav = self.augmentor(wav)
        return wav

    def _load_image(self, path: Optional[str]) -> Optional[Image.Image]:
        if not path:
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
            return None

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        task = row.get("task", "chat")
        is_audio_task = task in ("asr", "tts", "a2a")

        # Build target: if a separate `thinking` field exists, wrap it
        thinking = row.get("thinking", "")
        text_target = row.get("text_target", "")
        if thinking:
            text_target = f"<|think|>{thinking}<|/think|> {text_target}"

        return {
            "task":         task,
            "text_input":   row.get("text_input", ""),
            "text_target":  text_target,
            "image":        self._load_image(row.get("image_path")),
            "audio":        self._load_audio(row.get("audio_path"), augment=is_audio_task),
            "audio_target": self._load_audio(row.get("audio_target_path")),
        }


class WeightedMultiManifestDataset(Dataset):
    """Mixes multiple JSONL manifests with per-manifest sampling weights."""

    def __init__(
        self,
        manifest_paths: List[str],
        weights: Optional[List[float]] = None,
        augment_audio: bool = True,
    ) -> None:
        self.datasets = [
            MultiTaskManifestDataset(p, augment_audio=augment_audio) for p in manifest_paths
        ]
        weights = weights or [1.0] * len(self.datasets)

        self.index: List[tuple[int, int]] = [
            (di, li)
            for di, ds in enumerate(self.datasets)
            for li in range(len(ds))
        ]

        total_per_ds = [len(ds) for ds in self.datasets]
        self.sample_weights = torch.tensor(
            [weights[di] / total_per_ds[di] for di, _ in self.index],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        di, li = self.index[idx]
        return self.datasets[di][li]

    def make_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self.sample_weights.tolist(),
            num_samples=len(self),
            replacement=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskCollator:
    """Converts samples into model-ready tensors.

    Label masking: prompt tokens → -100, response tokens → real ids.
    Reasoning tasks (chat/code/math) whose target already contains <|think|>
    tokens are passed through as-is; the model learns to generate the full
    thinking trace + answer.
    """

    def __init__(
        self,
        tokenizer,
        vision_processor,
        audio_processor,
        max_text_len: int = 4096,
    ):
        self.tokenizer       = tokenizer
        self.vision_processor = vision_processor
        self.audio_processor  = audio_processor
        self.max_text_len     = max_text_len

    def _build_prompt(self, sample: dict) -> str:
        modal_prefix = ""
        if sample.get("image") is not None:
            modal_prefix += "<|image|> "
        if sample.get("audio") is not None:
            modal_prefix += "<|audio_in|> "
        return PROMPT_TEMPLATE.format(
            task=sample["task"],
            modal_prefix=modal_prefix,
            text_input=sample["text_input"],
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, Optional[torch.Tensor]]:
        all_input_ids: list[torch.Tensor] = []
        all_labels:    list[torch.Tensor] = []
        all_attn:      list[torch.Tensor] = []
        images: list = []
        audios: list = []

        eos = self.tokenizer.eos_token or ""

        for sample in batch:
            prompt = self._build_prompt(sample)
            target = sample["text_target"]

            prompt_ids = self.tokenizer(prompt,  add_special_tokens=True).input_ids
            target_ids = self.tokenizer(target + eos, add_special_tokens=False).input_ids

            # Keep at least 25% of capacity for the target
            max_p = self.max_text_len - min(len(target_ids), self.max_text_len // 4)
            prompt_ids = prompt_ids[-max_p:]               # truncate prompt from the left

            combined = (prompt_ids + target_ids)[: self.max_text_len]
            labels   = ([-100] * len(prompt_ids) + target_ids)[: self.max_text_len]

            all_input_ids.append(torch.tensor(combined, dtype=torch.long))
            all_labels.append(torch.tensor(labels,   dtype=torch.long))
            all_attn.append(torch.ones(len(combined),  dtype=torch.long))

            images.append(sample.get("image"))
            audios.append(sample.get("audio"))

        # ── Pad ──────────────────────────────────────────────────────────────
        pad_id  = self.tokenizer.pad_token_id or 0
        max_len = max(t.size(0) for t in all_input_ids)

        input_ids_b = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels_b    = torch.full((len(batch), max_len), -100,   dtype=torch.long)
        attn_b      = torch.zeros((len(batch), max_len),         dtype=torch.long)

        for i, (ids, lbl, attn) in enumerate(zip(all_input_ids, all_labels, all_attn)):
            n = ids.size(0)
            input_ids_b[i, :n] = ids
            labels_b[i,    :n] = lbl
            attn_b[i,      :n] = attn

        # ── Vision ───────────────────────────────────────────────────────────
        image_pixels = None
        valid_imgs = [img for img in images if img is not None]
        if valid_imgs:
            filled = [img if img is not None else valid_imgs[0] for img in images]
            image_pixels = self.vision_processor(images=filled, return_tensors="pt")["pixel_values"]

        # ── Audio ─────────────────────────────────────────────────────────────
        audio_features = None
        valid_auds = [a for a in audios if a is not None]
        if valid_auds:
            repl   = valid_auds[0]
            filled = [a if a is not None else repl for a in audios]
            af     = self.audio_processor(
                [a.numpy() for a in filled], sampling_rate=16000, return_tensors="pt"
            )
            audio_features = af.get("input_features")

        return {
            "input_ids":      input_ids_b,
            "attention_mask": attn_b,
            "labels":         labels_b,
            "image_pixels":   image_pixels,
            "audio_features": audio_features,
        }
