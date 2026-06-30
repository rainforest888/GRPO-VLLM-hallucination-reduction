# LLaVA Router Project — GRPO-Gated Attention for Hallucination Reduction

Based on LLaVA-1.5-7B + POPE benchmark.

## Setup

Model: `llava-hf/llava-1.5-7b-hf` (HuggingFace transformers)
Architecture: 32 LM layers, hidden=4096, 32 MHA heads, 576 vision tokens

## Strategy

Apply GRPO-trained router to dynamically select attention correction strategies
per layer to reduce object hallucination on POPE.
