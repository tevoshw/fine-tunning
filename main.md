# Fine-Tuning LLMs

Fine-tuning continues training a pre-trained model on smaller, specific data to change its **behavior** or specialize it for a task. It does not teach the model to *know* more, it teaches it to *act* differently using what it already knows.

## When to Use

| Need | Best path |
|------|-----------|
| New or updated knowledge | RAG (or CPT at scale) |
| Answers grounded in documents | RAG |
| Format control with few examples | Prompting / few-shot |
| Consistent behavior, style, format | **Fine-tuning** |
| Narrow task at scale, lower cost | **Fine-tuning** |

Rule of thumb: try prompting and RAG first.

## Types

**By objective**

- **SFT:** instruction → response pairs
- **Instruction / chat tuning:** many tasks or multi-turn dialogues
- **Task-specific / domain-adaptive:** one task or domain
- **Preference tuning:** RLHF, DPO
- **RLVR:** verifiable rewards (math, code)
- **Distillation:** small model learns from a larger one

**By method**

| Method | What is updated | Cost |
|--------|-----------------|------|
| **Full FT** | All weights | Highest |
| **Semi FT** | Part of the weights (last layers, FFN) | Medium |
| **LoRA** | Low-rank matrices, base frozen | Low |
| **QLoRA** | LoRA on a 4-bit quantized base (NF4) | Lowest |

Objective and method are independent: SFT can run with QLoRA, DPO with LoRA.

## How LoRA Works

Instead of learning a full `ΔW`, learn two small matrices:

```
ΔW = B · A        (W: d×k frozen, B: d×r, A: r×k, r << d,k)
h  = W·x + (α/r) · B·A·x
```

- `B` starts at zero, so training begins from the original behavior.
- Trains ~0.1% to 1% of parameters.
- Can be merged into `W` after training: no inference overhead.
- Learns less, forgets less than full FT.

## Practical Checklist

- **Data:** quality over quantity (1k to 10k strong examples), diverse, consistent, split into train/val/test, deduplicated.
- **Chat template:** identical in training and inference. Include the EOS token.
- **Loss masking:** compute loss on the response only.
- **Hyperparameters:**

| Parameter | Typical value |
|-----------|---------------|
| Learning rate | 1e-5 to 2e-5 (full), 1e-4 to 2e-4 (LoRA) |
| Epochs | 1 to 3 |
| LoRA rank (r) | 8 to 64 |
| LoRA alpha | r to 2r |
| Target modules | All linear layers (q, k, v, o + FFN) |

- **Efficiency:** gradient checkpointing, bf16, packing, FlashAttention.
- **Tools:** `transformers`, `peft`, `trl`, `bitsandbytes`, Unsloth, Axolotl.

## Risks

- Catastrophic forgetting
- Overfitting
- Induced hallucination (training on facts the model does not know)
- Noisy data
- Template mismatch
- Train/test leakage

## Quick Start

1. Define the task and metric.
2. Build 1k to 5k quality examples.
3. Run **QLoRA** on a 7B-8B model.
4. Evaluate against the original model and a prompting baseline.
5. Iterate on the data.