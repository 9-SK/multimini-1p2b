from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoProcessor,
    LlamaConfig,
    LlamaForCausalLM,
    WhisperModel,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiMiniConfig:
    # ── LLM backbone ────────────────────────────────────────────────────────
    vocab_size: int = 64000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    num_key_value_heads: int = 8       # GQA – halves KV-cache memory
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0       # Llama-3 style – good long-range extrapolation
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True   # requires flash-attn >= 2.x

    # ── Modal encoders ──────────────────────────────────────────────────────
    vision_backbone: str = "google/siglip-so400m-patch14-384"
    audio_backbone: str = "openai/whisper-large-v3"   # 1280-dim d_model
    modal_hidden_size: int = 1280
    projector_hidden_size: int = 4096
    projector_depth: int = 3

    freeze_vision_backbone: bool = True
    freeze_audio_backbone: bool = True


class MLPProjector(nn.Module):
    """Depth-configurable MLP projector with GELU activations and output LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(cur, hidden_dim), nn.GELU()]
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MultiMiniModel(nn.Module):
    """Unified multimodal CausalLM – H100 / BF16 optimised.

    Tasks
    -----
    chat   – text conversation (with optional <think> CoT reasoning)
    code   – coding / reasoning (same arch; task token differs)
    asr    – audio  → text
    tts    – text   → audio tokens
    a2a    – audio  → audio tokens
    i2t    – image  → text
    i2a    – image  → audio tokens
    """

    def __init__(self, cfg: MultiMiniConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── LLM backbone ──────────────────────────────────────────────────────
        attn_impl = "flash_attention_2" if cfg.use_flash_attention else "sdpa"
        llm_cfg = LlamaConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            rms_norm_eps=cfg.rms_norm_eps,
            tie_word_embeddings=cfg.tie_word_embeddings,
            use_cache=False,
            attn_implementation=attn_impl,
        )
        self.decoder = LlamaForCausalLM(llm_cfg)

        if cfg.gradient_checkpointing:
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ── Vision encoder (SigLIP) ──────────────────────────────────────────
        self.vision_processor = AutoProcessor.from_pretrained(cfg.vision_backbone)
        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_backbone, torch_dtype=torch.bfloat16
        )
        if cfg.freeze_vision_backbone:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # ── Audio encoder (Whisper-large-v3) ─────────────────────────────────
        whisper = WhisperModel.from_pretrained(cfg.audio_backbone, torch_dtype=torch.bfloat16)
        self.audio_encoder = whisper.encoder
        if cfg.freeze_audio_backbone:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

        # ── Projectors ───────────────────────────────────────────────────────
        vdim = getattr(self.vision_encoder.config, "hidden_size", cfg.modal_hidden_size)
        adim = getattr(self.audio_encoder.config, "d_model",      cfg.modal_hidden_size)

        self.vision_projector = MLPProjector(
            vdim, cfg.hidden_size, cfg.projector_hidden_size, depth=cfg.projector_depth
        )
        self.audio_projector = MLPProjector(
            adim, cfg.hidden_size, cfg.projector_hidden_size, depth=cfg.projector_depth
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, L, hidden_size) visual tokens."""
        out = self.vision_encoder(pixel_values=pixel_values)
        return self.vision_projector(out.last_hidden_state)

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """(B, T, hidden_size) audio tokens."""
        out = self.audio_encoder(input_features=input_features)
        return self.audio_projector(out.last_hidden_state)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_pixels: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ):
        tok_emb = self.decoder.get_input_embeddings()(input_ids)

        prefix_chunks: list[torch.Tensor] = []
        if image_pixels is not None:
            prefix_chunks.append(self.encode_image(image_pixels))
        if audio_features is not None:
            prefix_chunks.append(self.encode_audio(audio_features))

        if prefix_chunks:
            prefix = torch.cat(prefix_chunks, dim=1)
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            prefix_mask = torch.ones(
                attention_mask.size(0), prefix.size(1),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                ignore = torch.full(
                    (labels.size(0), prefix.size(1)), -100,
                    dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat([ignore, labels], dim=1)

            return self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
