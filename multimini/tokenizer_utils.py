from __future__ import annotations

from typing import Iterable

from transformers import AutoTokenizer


def build_tokenizer(
    tokenizer_name: str,
    extra_special_tokens: Iterable[str],
    num_audio_tokens: int = 2048,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    audio_tokens = [f"<|a_{i}|>" for i in range(num_audio_tokens)]
    special = list(extra_special_tokens) + audio_tokens
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    return tokenizer, audio_tokens
