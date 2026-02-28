# MultiMini-1.2B — H100 BF16 Multimodal + Reasoning

A unified multimodal + reasoning language model trained on H100 GPUs with BF16 precision.

## Capabilities

| Task token | Task |
|---|---|
| `<\|task_chat\|>` | Multi-turn conversation with chain-of-thought reasoning |
| `<\|task_code\|>` | Code generation + debugging with reasoning traces |
| `<\|task_math\|>` | Multi-step mathematical reasoning with `<\|think\|>` scratchpad |
| `<\|task_asr\|>` | Speech → text (audio encoder prefix + decoder) |
| `<\|task_tts\|>` | Text → audio tokens (EnCodec codec targets) |
| `<\|task_a2a\|>` | Audio → audio token transformation |
| `<\|task_i2t\|>` | Image → text / visual question answering |
| `<\|task_i2a\|>` | Image → audio token generation |

## Architecture

- **LLM backbone** — Llama-style decoder, 1.22 B parameters  
  22 layers · hidden 2048 · FFN 8192 · GQA (16 Q / 8 KV heads) · RoPE θ=500k · ctx 8192
- **Vision encoder** — SigLIP ViT-SO400M/14-384 (frozen) + 3-layer MLP projector
- **Audio encoder** — Whisper-large-v3 encoder (frozen) + 3-layer MLP projector
- **Precision** — BF16 end-to-end, Flash-Attention 2, ZeRO-3

## Reasoning

Thinking traces are wrapped with `<|think|>...<|/think|>` special tokens, inspired by
DeepSeek-R1. The model learns to produce explicit reasoning before its final answer on
`chat`, `code`, and `math` tasks.

## Quick start

```bash
# Install deps (Flash-Attention needs CUDA toolkit on PATH)
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Prepare all datasets
bash scripts/prepare_all.sh --cache_dir /data/hf_cache

# Launch training (single node, all H100s)
bash scripts/launch_h100.sh

# Or pass a single manifest directly
bash scripts/launch_h100.sh --manifest data/chat_reasoning.jsonl
```

## Directory layout

```
h100/
├── configs/
│   ├── model_1p2b.yaml             # model + training hyperparams
│   └── deepspeed_zero3_bf16.json   # ZeRO-3 H100 config
├── multimini/
│   ├── modeling.py                 # MultiMiniModel + MultiMiniConfig
│   ├── data.py                     # datasets, augmentation, collator
│   └── tokenizer_utils.py
├── scripts/
│   ├── prepare_all.sh              # run all dataset scripts
│   ├── prepare_stt.py              # LibriSpeech / GigaSpeech / CommonVoice / VoxPopuli
│   ├── prepare_tts.py              # LJSpeech / MLS / LibriTTS-R / VCTK
│   ├── prepare_vision.py           # NoCaps / LLaVA / COCO / Cauldron / VQAv2
│   ├── prepare_chat_reasoning.py   # OpenHermes / SlimOrca / UltraChat / TULU-3
│   ├── prepare_code.py             # OpenThoughts / Magicoder / OpenCodeInterpreter
│   ├── prepare_math.py             # NuminaMath / MATH / Orca-Math / GSM8K
│   └── launch_h100.sh              # multi-GPU DeepSpeed launcher
├── train.py                        # main training entry point
├── accelerate_zero3_bf16.yaml
└── requirements.txt
```

## Datasets used

### Text / Chat / Reasoning
| Dataset | Size | Notes |
|---|---|---|
| teknium/OpenHermes-2.5 | 1M | GPT-4 sourced instruction pairs |
| Open-Orca/SlimOrca | 518k | Orca reasoning traces, deduped |
| HuggingFaceH4/ultrachat_200k | 200k | Multi-turn conversations |
| allenai/tulu-3-sft-mixture | 940k | Diverse SFT superset |
| nvidia/HelpSteer2 | 21k | Helpfulness-ranked pairs |
| argilla/magpie-ultra-v0.1 | 50k | Self-play ultra quality |

### Code + Reasoning
| Dataset | Size | Notes |
|---|---|---|
| open-thoughts/OpenThoughts-114k | 114k | R1-distilled code+math reasoning |
| m-a-p/OpenCodeInterpreter-OS | 68k | Execution + explanation pairs |
| ise-uiuc/Magicoder-OSS-Instruct-75K | 75k | OS-seeded code instruction |
| bigcode/self-oss-instruct-sc2-exec-filter-50k | 50k | Exec-verified self-instruct |

### Math + Reasoning
| Dataset | Size | Notes |
|---|---|---|
| AI-MO/NuminaMath-CoT | 860k | Competition math + CoT |
| lighteval/MATH | 12.5k | AMC/AIME competition |
| microsoft/orca-math-word-problems-200k | 200k | GPT-4 word problems |
| qwedsacf/grade-school-math (GSM8K) | 7.5k | Multi-step grade-school math |

### STT / ASR (audio → text)
| Dataset | Size | Notes |
|---|---|---|
| openslr/librispeech_asr | 960h | Clean + other splits |
| speechcolab/gigaspeech (xl) | 10k h | Curated web speech |
| mozilla-foundation/common_voice_17_0 | ~2k h | Multi-accent crowd-sourced |
| facebook/voxpopuli | 400h EN | Parliamentary speech |

### TTS (text → audio)
| Dataset | Size | Notes |
|---|---|---|
| keithito/lj_speech | 24h | Single-speaker studio |
| facebook/multilingual_librispeech (en) | 44.5h | Multi-speaker clean |
| blabble-io/libritts (clean) | 245h | Studio-cleaned LibriSpeech |
| WillHeld/vctk | 44h | 109-speaker British English |

### Vision / Image → Text
| Dataset | Size | Notes |
|---|---|---|
| merve/coco_captions | 118k images | 5 captions each |
| liuhaotian/LLaVA-Instruct-150K | 150k | Visual instruction following |
| HuggingFaceM4/NoCaps | 100k | Open-domain captioning |
| HuggingFaceM4/the_cauldron | 50M+ | Multi-task VQA superset |
| Multimodal-Fatima/VQAv2_train | 443k | Visual QA |

## Training recipe

| Setting | Value |
|---|---|
| Batch per GPU | 8 |
| Grad accum | 8 |
| Effective batch (8×H100) | 512 |
| LR | 1.5e-4 cosine → 1.5e-5 |
| Warmup | 2 000 steps |
| Max steps | 500 000 |
| Precision | BF16 |
| ZeRO stage | 3 (no offload) |
| Flash-Attention | 2 |
| Gradient checkpointing | ✓ |
| `torch.compile` | ✓ reduce-overhead |

## Sampling weights

Tasks are mixed by the `WeightedMultiManifestDataset` with weights:

```
chat 3.0 · code 2.5 · math 2.0 · asr 1.5 · tts 1.0 · image 1.0 · a2a 0.5
```
