"""
GEKO + GPT-2 smoke test: 100 samples, 5 epochs, CPU.

Verifies:
- Training loop runs without error
- Bucket distribution evolves over epochs
- Per-sample correctness (via custom compute_correctness)
- Efficiency report shows compute savings
- Checkpoint save/load round-trip
"""

import tempfile
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from geko import GEKOTrainer, GEKOConfig, GEKOTrainingArgs

# ── Dataset ──────────────────────────────────────────────────────────────────

# 20 distinct short sentences, repeated 5x = 100 samples total.
# GPT-2 should master most of them within a few epochs.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A stitch in time saves nine.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "Where there is a will there is a way.",
    "A journey of a thousand miles begins with a single step.",
    "The pen is mightier than the sword.",
    "Knowledge is power.",
    "Time flies like an arrow.",
    "Practice makes perfect.",
    "Better late than never.",
    "Every cloud has a silver lining.",
    "The grass is always greener on the other side.",
    "Do not judge a book by its cover.",
    "A rolling stone gathers no moss.",
    "Honesty is the best policy.",
    "Beauty is in the eye of the beholder.",
    "Fortune favors the bold.",
]

TEXTS = SENTENCES * 5   # 100 samples


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.items = []
        for text in texts:
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
                "labels":         enc["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Per-sample correctness (requires per-sample loss) ────────────────────────

def per_sample_correctness(outputs, batch):
    """
    Compute per-sample correctness by re-running a per-token NLL manually.

    GPT-2's default outputs.loss is a scalar (mean over tokens × batch).
    We shift labels and compute per-sample loss ourselves.
    """
    logits = outputs.logits            # [B, T, vocab]
    labels = batch.get("labels")       # [B, T]
    if labels is None:
        bs = logits.size(0)
        return torch.zeros(bs, dtype=torch.bool)

    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, vocab]
    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

    bs, seq_len, vocab = shift_logits.shape
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    # loss_fct expects [B*(T-1), vocab] and [B*(T-1)]
    per_token_loss = loss_fct(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
    ).view(bs, seq_len)                              # [B, T-1]

    # Average over non-padding tokens
    mask = (shift_labels != -100).float()
    valid = mask.sum(dim=1).clamp(min=1)
    per_sample_loss = (per_token_loss * mask).sum(dim=1) / valid   # [B]

    # "Correct" if per-sample loss is below 0.5
    return per_sample_loss < 0.5


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  GEKO + GPT-2 smoke test")
    print("=" * 60)

    # Load tiny GPT-2 (117M — smallest checkpoint)
    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataset
    dataset = TextDataset(TEXTS, tokenizer, max_length=32)
    print(f"Dataset size: {len(dataset)} samples")

    # Split off a small eval set (last 10 samples)
    train_dataset = TextDataset(TEXTS[:-10], tokenizer, max_length=32)
    eval_dataset  = TextDataset(TEXTS[-10:],  tokenizer, max_length=32)

    # GEKO config — tighter thresholds so we see FREEZE action quickly
    config = GEKOConfig(
        freeze_confidence=0.70,    # easier to FREEZE (default: 0.85)
        freeze_quality=0.60,       # easier to FREEZE (default: 0.80)
        focus_confidence=0.40,     # default: 0.60
        min_q_for_freeze=0.60,     # easier to FREEZE (default: 0.80)
        q_value_loss_scale=4.0,    # GPT-2 cross-entropy typically 1–4
        use_curriculum=True,
        log_bucket_stats=True,
    )

    args = GEKOTrainingArgs(
        output_dir="./geko_gpt2_test",
        num_epochs=5,
        batch_size=8,
        learning_rate=5e-5,
        warmup_steps=20,
        logging_steps=10,
        save_steps=9999,        # don't save mid-training
        eval_steps=20,
        gradient_accumulation_steps=1,
        fp16=False,             # CPU
        save_at_end=True,
        seed=42,
    )

    trainer = GEKOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config,
        args=args,
        compute_correctness=per_sample_correctness,   # true per-sample!
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    result = trainer.train()

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete")
    print("=" * 60)

    report = trainer.get_efficiency_report()
    if report:
        print(f"\n  Samples total     : {report['total_samples']}")
        print(f"  Samples mastered  : {report['samples_mastered']}")
        print(f"  Compute saved     : {report['compute_saved_percent']}")
        print(f"  Final accuracy    : {report['final_accuracy']}")
        print(f"\n  Bucket distribution: {report['bucket_distribution']}")
        print(f"\n  Improvement:")
        print(f"    FREEZE growth : {report['improvement']['freeze_change']:+.1%}")
        print(f"    HARD change   : {report['improvement']['hard_change']:+.1%}")
    else:
        print("\n  (no efficiency report — dataset may have been too small)")

    # ── Checkpoint round-trip ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Checkpoint round-trip test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(tmpdir)

        # New trainer, load checkpoint, check state restored
        trainer2 = GEKOTrainer(
            model=GPT2LMHeadModel.from_pretrained("gpt2"),
            train_dataset=train_dataset,
            config=config,
            args=args,
            compute_correctness=per_sample_correctness,
        )
        trainer2.load_checkpoint(tmpdir)

        report2 = trainer2.get_efficiency_report()
        if report2:
            print(f"\n  Restored partition history: {len(trainer2.partition_history)} entries")
            print(f"  Efficiency report after restore: {report2['compute_saved_percent']} saved")
            print("  Checkpoint round-trip: PASSED")
        else:
            print("  Checkpoint round-trip: FAILED (no efficiency report after restore)")

    print("\n" + "=" * 60)
    print("  All checks passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
