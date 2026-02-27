"""
GEKO v0.3.0 — GPT-2 on open-r1/OpenR1-Math-220k (full dataset)

Trains GPT-2 on all 220k R1 math reasoning samples using GEKO's
intelligent sample selection. Demonstrates all v0.3.0 features:
  - Dynamic dataset pruning (mastered samples permanently removed)
  - BF16 / FP16 mixed precision
  - Gradient checkpointing
  - torch.compile (PyTorch 2.0+)
  - Non-blocking fast DataLoader
  - Stable bucket caching

Requirements:
    pip install transformers datasets gekolib tqdm

Usage:
    python train_r1_gpt2.py
"""

import sys
import os

# Force unbuffered output so every print shows immediately
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, closefd=False)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, closefd=False)

print("=" * 60)
print("  GEKO v0.3.0 + GPT-2 on OpenR1-Math-220k")
print("=" * 60)
print("\n[Step 1/6] Importing libraries...")
sys.stdout.flush()

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

print("  torch         OK")
sys.stdout.flush()

from transformers import GPT2LMHeadModel, GPT2Tokenizer
print("  transformers  OK")
sys.stdout.flush()

from datasets import load_dataset
print("  datasets      OK")
sys.stdout.flush()

from geko import GEKOTrainer, GEKOConfig, GEKOTrainingArgs
print("  geko          OK")
sys.stdout.flush()


# ── Config ────────────────────────────────────────────────────────────────────

MAX_LENGTH   = 512
BATCH_SIZE   = 16
NUM_EPOCHS   = 3
LR           = 3e-5
OUTPUT_DIR   = "./geko_r1_output"

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16

print(f"\n  Device : {DEVICE.upper()}")
print(f"  BF16   : {USE_BF16}")
print(f"  FP16   : {USE_FP16}")
sys.stdout.flush()


# ── Dataset ───────────────────────────────────────────────────────────────────

class R1MathDataset(TorchDataset):
    """Wraps OpenR1-Math-220k as a PyTorch dataset for GEKO."""

    def __init__(self, hf_dataset, tokenizer, max_length: int = 512, desc: str = "Tokenizing"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        total = len(hf_dataset)
        bar = tqdm(
            enumerate(hf_dataset),
            total=total,
            desc=f"  {desc}",
            unit="sample",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for i, example in bar:
            text = self._format(example)
            enc = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.items.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         enc["input_ids"].squeeze(0).clone(),
            })

            if (i + 1) % 5_000 == 0:
                bar.set_postfix(done=f"{i+1:,}/{total:,}")

        bar.close()
        print(f"  Done — {len(self.items):,} samples ready.")
        sys.stdout.flush()

    def _format(self, example: dict) -> str:
        problem  = example.get("problem",  example.get("question", ""))
        solution = example.get("solution", example.get("answer",   ""))
        return f"Problem: {problem}\n\nSolution: {solution}"

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


# ── Per-sample correctness ────────────────────────────────────────────────────

def per_sample_correctness(outputs, batch):
    """
    True per-sample correctness via manually computed per-token NLL.
    GPT-2's outputs.loss is a scalar (batch mean). We shift labels and
    compute per-sample loss ourselves so GEKO can bucket at sample level.
    """
    logits = outputs.logits          # [B, T, vocab]
    labels = batch.get("labels")
    if labels is None:
        return torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    bs, seq_len, vocab = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(bs, seq_len)

    mask  = (shift_labels != -100).float()
    valid = mask.sum(dim=1).clamp(min=1)
    per_sample_loss = (per_token_loss * mask).sum(dim=1) / valid

    # "Correct" = loss < 1.5 nats ≈ perplexity < 4.5
    return per_sample_loss < 1.5


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Step 2: Model & tokenizer ─────────────────────────────────────────────
    print("\n[Step 2/6] Loading GPT-2 model & tokenizer...")
    sys.stdout.flush()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GPT-2 loaded — {n_params:,} parameters")
    sys.stdout.flush()

    # ── Step 3: Download dataset ───────────────────────────────────────────────
    print("\n[Step 3/6] Downloading open-r1/OpenR1-Math-220k...")
    print("  (HuggingFace will show a progress bar below)")
    sys.stdout.flush()

    raw = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    n_total = len(raw)
    print(f"  Downloaded — {n_total:,} samples")
    sys.stdout.flush()

    # ── Step 4: Tokenize ──────────────────────────────────────────────────────
    print(f"\n[Step 4/6] Tokenizing (max_length={MAX_LENGTH})...")
    sys.stdout.flush()

    train_raw = raw.select(range(1000))
    eval_raw  = raw.select(range(1000, 1100))

    train_dataset = R1MathDataset(train_raw, tokenizer, max_length=MAX_LENGTH, desc="Train")
    eval_dataset  = R1MathDataset(eval_raw,  tokenizer, max_length=MAX_LENGTH, desc="Eval ")

    # ── Step 5: GEKO config ───────────────────────────────────────────────────
    print("\n[Step 5/6] Configuring GEKO...")
    sys.stdout.flush()

    config = GEKOConfig(
        freeze_confidence  = 0.85,
        freeze_quality     = 0.80,
        min_q_for_freeze   = 0.70,
        focus_confidence   = 0.50,
        q_value_loss_scale = 5.0,
        use_curriculum     = True,
        log_bucket_stats   = True,
        prune_frozen_after = 2,
        repartition_every  = 1,
    )

    args = GEKOTrainingArgs(
        output_dir                  = OUTPUT_DIR,
        num_epochs                  = NUM_EPOCHS,
        batch_size                  = BATCH_SIZE,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_steps                = 50,
        logging_steps               = 10,
        save_steps                  = 9999,
        eval_steps                  = 30,
        max_grad_norm               = 1.0,
        gradient_accumulation_steps = 4,
        fp16                        = USE_FP16,
        bf16                        = USE_BF16,
        gradient_checkpointing      = True,
        compile_model               = True,
        seed                        = 42,
        save_at_end                 = True,
    )

    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}  (effective: {BATCH_SIZE * 4} with grad accum)")
    print(f"  Learning rate : {LR}")
    print(f"  Train samples : {len(train_dataset):,}")
    print(f"  Eval  samples : {len(eval_dataset):,}")
    sys.stdout.flush()

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    print("\n[Step 6/6] Starting GEKO training...\n")
    sys.stdout.flush()

    trainer = GEKOTrainer(
        model               = model,
        train_dataset       = train_dataset,
        eval_dataset        = eval_dataset,
        tokenizer           = tokenizer,
        config              = config,
        args                = args,
        compute_correctness = per_sample_correctness,
    )

    result = trainer.train()

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    sys.stdout.flush()

    report = trainer.get_efficiency_report()
    if report:
        print(f"\n  Samples total     : {report['total_samples']:,}")
        print(f"  Samples mastered  : {report['samples_mastered']:,}")
        print(f"  Samples pruned    : {trainer.get_pruned_count():,}")
        print(f"  Compute saved     : {report['compute_saved_percent']}")
        print(f"  Final accuracy    : {report['final_accuracy']}")
        print(f"\n  Bucket distribution:")
        print(f"    {report['bucket_distribution']}")
        print(f"\n  Improvement:")
        print(f"    FREEZE growth : {report['improvement']['freeze_change']:+.1%}")
        print(f"    HARD change   : {report['improvement']['hard_change']:+.1%}")

    print(f"\n  Avg loss          : {result['total_loss']:.4f}")
    print(f"  Samples trained on: {result['samples_trained']:,}")
    print(f"\n  Checkpoint saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 60 + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
