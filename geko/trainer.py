"""
GEKO Trainer

A drop-in training wrapper that works with ANY language model.
Like LoRA wraps models for efficient fine-tuning,
GEKO wraps training for efficient learning.

Key Features:
- Works with any HuggingFace model
- Automatic sample partitioning (FREEZE/LIGHT/FOCUS/HARD)
- Mountain Curriculum for optimal learning progression
- Per-sample Q-value tracking
- Automatic early stopping when dataset is mastered
"""

import os
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from collections import deque
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import json
from tqdm import tqdm

from .core import Bucket, SampleState, GEKOConfig
from .partitioner import SamplePartitioner, PartitionStats
from .curriculum import MountainCurriculum, CurriculumPhase


class GEKODataset(Dataset):
    """
    Wraps any Dataset to inject a global `sample_id` into each item.

    This ensures the trainer can track per-sample learning state correctly
    regardless of batch ordering or shuffling.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        # Probe the first item immediately so users get a clear error at
        # construction time rather than a cryptic KeyError inside the training loop.
        if len(dataset) > 0:
            sample = dataset[0]
            if not isinstance(sample, dict):
                raise TypeError(
                    f"GEKOTrainer requires your dataset's __getitem__ to return a dict "
                    f"(got {type(sample).__name__}). Each item must include at minimum an "
                    f"'input_ids' key. Wrap your dataset so it returns a dict, e.g.:\n\n"
                    f"    def __getitem__(self, idx):\n"
                    f"        x, y = self.data[idx]\n"
                    f"        return {{'input_ids': x, 'labels': y}}"
                )
            if 'input_ids' not in sample:
                raise TypeError(
                    "Each dataset item must include an 'input_ids' key. "
                    "Your dataset's __getitem__ should return a dict with at least 'input_ids', e.g.:\n\n"
                    "    def __getitem__(self, idx):\n"
                    "        x, y = self.data[idx]\n"
                    "        return {'input_ids': x, 'labels': y}"
                )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        item['sample_id'] = str(idx)
        return item


@dataclass
class GEKOTrainingArgs:
    """Training arguments for GEKOTrainer."""
    output_dir: str = "./geko_output"
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    max_grad_norm: float = 1.0

    # Precision — fp16 auto-enables on CUDA; bf16 preferred on A100/H100 (no overflow)
    fp16: bool = field(default_factory=lambda: torch.cuda.is_available())
    bf16: bool = False  # If True and fp16 also True, bf16 wins (no GradScaler needed)

    gradient_accumulation_steps: int = 1

    # Gradient checkpointing: trade compute for memory (~4x activation memory reduction)
    gradient_checkpointing: bool = False

    # torch.compile: 20-50% speedup on PyTorch 2.0+ (JIT fusion of model ops)
    compile_model: bool = False

    # 8-bit Adam: optimizer states int8 instead of fp32 → 2x optimizer memory reduction
    # Requires: pip install bitsandbytes  (or pip install gekolib[bnb])
    use_8bit_optimizer: bool = False

    # DataLoader settings
    # -1 = auto-detect: min(4, cpu_count) on Linux/Windows, 0 on macOS (fork issues)
    dataloader_num_workers: int = -1
    dataloader_persistent_workers: bool = True   # Keep workers alive between epochs
    dataloader_prefetch_factor: int = 2          # Batches to prefetch per worker

    seed: int = 42
    # Set False if you want to manage checkpointing yourself
    save_at_end: bool = True


class GEKOTrainer:
    """
    GEKO-enhanced trainer for any language model.

    Wraps standard training with GEKO's intelligent sample selection:
    1. Evaluates model confidence on each sample
    2. Partitions into FREEZE/LIGHT/FOCUS/HARD buckets
    3. Prioritizes training on HARD samples (confident-wrong)
    4. Skips FREEZE samples (already mastered)
    5. Follows Mountain Curriculum for optimal progression

    Usage:
        from geko import GEKOTrainer, GEKOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        trainer = GEKOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            config=GEKOConfig(),
        )

        # Train with GEKO optimization
        trainer.train()

        # Check efficiency gains
        print(trainer.get_efficiency_report())
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        tokenizer: Optional[Any] = None,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[GEKOConfig] = None,
        args: Optional[GEKOTrainingArgs] = None,
        compute_confidence: Optional[Callable] = None,
        compute_correctness: Optional[Callable] = None,
        lora_config: Optional[Any] = None,
    ):
        """
        Initialize GEKO Trainer.

        Args:
            model: Any PyTorch model (HuggingFace, custom, etc.)
            train_dataset: Training dataset (must return dicts with at least 'input_ids')
            tokenizer: Optional tokenizer (stored for user convenience; not called internally)
            eval_dataset: Optional evaluation dataset (same dict structure as training; see README)
            config: GEKO configuration
            args: Training arguments
            compute_confidence: Optional function(outputs, batch) → Tensor[batch_size].
                Defaults to max softmax probability.
            compute_correctness: Optional function(outputs, batch) → BoolTensor[batch_size].
                Default: per-sample when model returns 1D loss, else batch-level (one value per batch).
                When the model returns a scalar loss, override this for true per-sample bucketing.
            lora_config: Optional peft.LoraConfig. If provided, wraps model with LoRA adapters
                before training. Requires: pip install peft
        """
        # Apply LoRA before anything else (changes model structure)
        if lora_config is not None:
            from .peft_utils import apply_lora
            model = apply_lora(model, lora_config)

        self.model = model
        self.tokenizer = tokenizer  # stored for user convenience; not called internally
        # Wrap dataset to inject global sample IDs into each batch
        self.train_dataset = GEKODataset(train_dataset)
        self.eval_dataset = eval_dataset
        if eval_dataset is not None and len(eval_dataset) > 0:
            sample = eval_dataset[0]
            if not isinstance(sample, dict):
                raise TypeError(
                    "eval_dataset must return dicts (same structure as training). "
                    f"Got {type(sample).__name__}. Ensure __getitem__ returns a dict with model input keys."
                )
        self.config = config or GEKOConfig()
        self.args = args or GEKOTrainingArgs()

        # Custom evaluation functions
        self.compute_confidence = compute_confidence or self._default_confidence
        self.compute_correctness = compute_correctness or self._default_correctness

        # GEKO components
        self.partitioner = SamplePartitioner(self.config)
        self.curriculum = MountainCurriculum(
            total_samples=len(self.train_dataset) * self.args.num_epochs,
            config=self.config
        ) if self.config.use_curriculum else None

        # Sample states (tracks learning progress per sample)
        self.sample_states: Dict[str, SampleState] = {}
        self._init_sample_states()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.partition_history: List[PartitionStats] = []
        self._warned_batch_level_correctness = False
        self._last_bucket_distribution: Optional[Tuple[int, int, int, int]] = None
        self._cached_dataloader: Optional[DataLoader] = None
        self._pruned_count: int = 0

        # Device
        self.device = next(model.parameters()).device

    @staticmethod
    def _get_batch_size(batch: dict) -> int:
        """Infer batch size from the first tensor in the batch."""
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1

    def _init_sample_states(self):
        """Initialize sample states for all samples."""
        for idx in range(len(self.train_dataset)):
            sample_id = str(idx)
            self.sample_states[sample_id] = SampleState(
                sample_id=sample_id,
                bucket=Bucket.FOCUS,  # Start in FOCUS (assume uncertain)
                q_value=0.5,
            )

    def _default_confidence(
        self,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Default confidence: max softmax probability over the vocabulary.

        Extracted directly from the training forward pass outputs — no extra
        forward pass needed.

        Returns a 1-D tensor of shape [batch_size].
        Override compute_confidence with a function(outputs, batch) → Tensor
        to use a custom confidence metric.
        """
        if hasattr(outputs, 'logits'):
            # .float() avoids half-precision overflow in softmax
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            max_probs = probs.max(dim=-1).values
            # Average across sequence length → shape [batch_size]
            return max_probs.mean(dim=-1) if max_probs.dim() > 1 else max_probs
        # Fallback if model doesn't expose logits (e.g. custom architectures)
        batch_size = self._get_batch_size(batch)
        return torch.full((batch_size,), 0.5, device=self.device)

    def _default_correctness(
        self,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Default correctness: per-sample when model returns 1D loss, else batch-level.

        When the model returns a scalar loss, every sample in the batch gets the same
        correctness (batch-level). For true per-sample bucketing, override
        compute_correctness with a function(outputs, batch) → BoolTensor[batch_size].

        Returns a bool Tensor of shape [batch_size].
        """
        batch_size = self._get_batch_size(batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        if loss.dim() == 1 and loss.numel() == batch_size:
            return (loss < threshold).to(dtype=torch.bool, device=self.device)
        # Batch-level fallback: one value for whole batch
        if not self._warned_batch_level_correctness:
            print(
                "[GEKO] Using batch-level correctness (model returns scalar loss). "
                "For per-sample bucketing, override compute_correctness; see API Reference."
            )
            self._warned_batch_level_correctness = True
        loss_val = loss.mean().item() if loss.dim() > 0 else loss.item()
        return torch.tensor(
            [loss_val < threshold] * batch_size,
            dtype=torch.bool,
            device=self.device,
        )

    def _get_sample_weights(self) -> List[float]:
        """Get sampling weights based on bucket assignments."""
        weights = []
        for idx in range(len(self.train_dataset)):
            sample_id = str(idx)
            state = self.sample_states[sample_id]

            # Exclude samples that have been trained on too many times
            if state.times_seen >= self.config.max_times_seen:
                weights.append(0.0)
                continue

            weight = self.config.get_bucket_weight(state.bucket)

            # Apply curriculum adjustment
            if self.curriculum:
                curr_weights = self.curriculum.get_current_weights()
                bucket_idx = {Bucket.HARD: 0, Bucket.FOCUS: 1, Bucket.LIGHT: 2}.get(state.bucket, -1)
                if bucket_idx >= 0:
                    weight = curr_weights[bucket_idx]

            # FREEZE samples get weight 0 — excluded from sampling
            weights.append(float(weight))

        return weights

    def _resolve_num_workers(self) -> int:
        """Auto-detect optimal DataLoader num_workers."""
        if self.args.dataloader_num_workers >= 0:
            return self.args.dataloader_num_workers
        # macOS has multiprocessing fork issues with DataLoader workers
        if platform.system() == 'Darwin':
            return 0
        return min(4, os.cpu_count() or 1)

    def _create_dataloader(self, weighted: bool = True) -> DataLoader:
        """Create dataloader with GEKO-weighted sampling and fast settings."""
        pin = (self.device.type == 'cuda')
        num_workers = self._resolve_num_workers()
        persistent = num_workers > 0 and self.args.dataloader_persistent_workers
        prefetch = self.args.dataloader_prefetch_factor if num_workers > 0 else None

        if weighted:
            weights = self._get_sample_weights()
            # If every sample is FREEZE, all weights are 0 — WeightedRandomSampler would
            # raise ValueError. Fall back to uniform; training will auto-stop next repartition.
            if sum(weights) == 0:
                weights = [1.0] * len(self.train_dataset)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
            )

    def _update_sample_states(
        self,
        sample_ids: List[str],
        losses: torch.Tensor,
        confidences: torch.Tensor,
        corrects: torch.Tensor
    ):
        """Update sample states after a training step."""
        lr = self.config.q_value_lr
        loss_scale = self.config.q_value_loss_scale
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.sample_states:
                loss_val = losses[i].item() if losses.dim() > 0 else losses.item()
                conf_val = confidences[i].item() if confidences.dim() > 0 else confidences.item()
                corr_val = bool(corrects[i].item() if corrects.dim() > 0 else corrects.item())
                self.sample_states[sample_id].update(
                    loss=loss_val,
                    confidence=conf_val,
                    correct=corr_val,
                    epoch=self.current_epoch,
                    lr=lr,
                    loss_scale=loss_scale,
                )

    def partition_samples(self) -> PartitionStats:
        """
        Re-partition all samples based on current model performance.

        Call this at the start of each epoch to update bucket assignments.
        Also handles consecutive_frozen_epochs tracking and dataset pruning.
        """
        stats = self.partitioner.partition(self.sample_states, self.current_epoch)
        self.partition_history.append(stats)

        if self.config.log_bucket_stats:
            print(f"\n[GEKO] Epoch {self.current_epoch} Partition: {stats}")

        # Update consecutive_frozen_epochs and prune if configured
        to_prune = []
        for sample_id, state in self.sample_states.items():
            if state.bucket == Bucket.FREEZE:
                state.consecutive_frozen_epochs += 1
            else:
                state.consecutive_frozen_epochs = 0

            if (self.config.prune_frozen_after > 0 and
                    state.consecutive_frozen_epochs >= self.config.prune_frozen_after):
                to_prune.append(sample_id)

        if to_prune:
            for sample_id in to_prune:
                del self.sample_states[sample_id]
            self._pruned_count += len(to_prune)
            print(
                f"[GEKO] Pruned {len(to_prune)} samples "
                f"(frozen for {self.config.prune_frozen_after}+ epochs). "
                f"Active dataset: {len(self.sample_states)} samples."
            )

        return stats

    def get_pruned_count(self) -> int:
        """Return the total number of samples permanently pruned so far."""
        return self._pruned_count

    def train(self):
        """
        Main training loop with GEKO optimization.

        Returns:
            Dict with training results and efficiency metrics
        """
        # Resolve precision mode
        use_bf16 = self.args.bf16
        use_fp16 = self.args.fp16 and not use_bf16
        if self.args.bf16 and self.args.fp16:
            print("[GEKO] Both bf16 and fp16 set — bf16 takes priority.")
        precision_str = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")

        print(
            f"\n{'='*55}\n"
            f"  GEKO Training\n"
            f"{'='*55}\n"
            f"  Samples           : {len(self.train_dataset)}\n"
            f"  Epochs            : {self.args.num_epochs}\n"
            f"  Batch size        : {self.args.batch_size}\n"
            f"  Device            : {self.device}\n"
            f"  Precision         : {precision_str}\n"
            f"  Grad accum        : {self.args.gradient_accumulation_steps}\n"
            f"  Grad checkpointing: {'ON' if self.args.gradient_checkpointing else 'OFF'}\n"
            f"  torch.compile     : {'ON' if self.args.compile_model else 'OFF'}\n"
            f"  8-bit optimizer   : {'ON' if self.args.use_8bit_optimizer else 'OFF'}\n"
            f"  DataLoader workers: {self._resolve_num_workers()}\n"
            f"  Warmup steps      : {self.args.warmup_steps}\n"
            f"  Curriculum        : {'ON' if self.curriculum else 'OFF'}\n"
            f"  Config            :\n{self.config}\n"
            f"{'='*55}\n"
        )

        # Seed for reproducibility
        torch.manual_seed(self.args.seed)

        # Gradient checkpointing (trade activation memory for recompute)
        if self.args.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("[GEKO] Gradient checkpointing enabled")
            else:
                print("[GEKO] Warning: model does not support gradient_checkpointing_enable(), skipping")

        # torch.compile (PyTorch 2.0+ JIT fusion — 20-50% speedup)
        if self.args.compile_model:
            if hasattr(torch, 'compile'):
                print("[GEKO] Compiling model with torch.compile (first batch will be slow)...")
                self.model = torch.compile(self.model)
            else:
                print("[GEKO] Warning: torch.compile not available (requires PyTorch 2.0+), skipping")

        self.model.train()

        # Setup optimizer (standard or 8-bit)
        if self.args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
                print("[GEKO] Using 8-bit AdamW optimizer")
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for use_8bit_optimizer=True. Install it with:\n"
                    "    pip install bitsandbytes\n"
                    "or:\n"
                    "    pip install gekolib[bnb]"
                )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        # Linear LR warmup: ramps from 0 → base_lr over warmup_steps optimizer steps
        def _lr_lambda(current_optimizer_step: int) -> float:
            if current_optimizer_step < self.args.warmup_steps:
                return float(current_optimizer_step) / float(max(1, self.args.warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

        # Setup mixed precision
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        # bf16 doesn't need GradScaler (no overflow risk); fp16 does
        if use_bf16:
            scaler = None
            autocast_dtype = torch.bfloat16
        elif use_fp16:
            scaler = torch.amp.GradScaler(device_type)
            autocast_dtype = torch.float16
        else:
            scaler = None
            autocast_dtype = None

        use_amp = use_bf16 or use_fp16

        # Training loop
        total_loss = 0
        samples_trained = 0
        grad_accum = self.args.gradient_accumulation_steps

        optimizer.zero_grad()  # initialise before first accumulation window

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch

            # Re-partition at start of each epoch
            if epoch % self.config.repartition_every == 0:
                stats = self.partition_samples()

                # Check for early stopping
                if self.partitioner.should_stop_early(stats):
                    print(f"\n[GEKO] Early stopping: {stats.freeze_ratio:.1%} samples mastered!")
                    break

            # Skip epoch if no trainable samples (all FREEZE or max_times_seen)
            weights = self._get_sample_weights()
            if sum(weights) == 0:
                print(
                    "\n[GEKO] All samples are mastered or at max_times_seen; "
                    "skipping this epoch. GEKO will stop after the next partition check."
                )
                continue

            # Create weighted dataloader — only rebuild if bucket distribution changed >5%
            total = len(self.sample_states) or 1
            current_dist = (stats.freeze_count, stats.light_count, stats.focus_count, stats.hard_count)
            dist_changed = True
            if self._last_bucket_distribution is not None and self._cached_dataloader is not None:
                dist_changed = any(
                    abs(current_dist[i] - self._last_bucket_distribution[i]) / total > 0.05
                    for i in range(4)
                )
                if not dist_changed:
                    print("[GEKO] Bucket distribution stable — reusing dataloader")
            if dist_changed:
                self._cached_dataloader = self._create_dataloader(weighted=True)
                self._last_bucket_distribution = current_dist
            dataloader = self._cached_dataloader

            # Epoch training
            epoch_loss = 0
            epoch_samples = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                # Move tensors to device (non_blocking overlaps transfer with compute)
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Extract sample IDs (always present due to GEKODataset wrapper).
                # Fallback uses actual batch size — NOT args.batch_size — so the last
                # (potentially smaller) batch gets the right number of IDs.
                sample_ids = batch.pop('sample_id', None)
                if sample_ids is None:
                    actual_size = self._get_batch_size(batch)
                    sample_ids = [str(i) for i in range(actual_size)]

                # Forward pass
                batch_size = self._get_batch_size(batch)
                if use_amp:
                    with torch.amp.autocast(device_type, dtype=autocast_dtype):
                        outputs = self.model(**batch)
                        loss_raw = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    outputs = self.model(**batch)
                    loss_raw = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                # Scalar loss for backward; keep per-sample when model returns 1D loss
                if loss_raw.dim() == 1 and loss_raw.numel() == batch_size:
                    loss_for_backward = loss_raw.mean()
                else:
                    loss_for_backward = loss_raw.mean() if loss_raw.dim() > 0 else loss_raw

                # Scale loss for gradient accumulation
                loss = loss_for_backward / grad_accum

                # Backward pass (gradients accumulate across micro-batches)
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Determine if this batch completes an accumulation window
                is_last_batch = (batch_idx + 1) == len(dataloader)
                should_step = ((batch_idx + 1) % grad_accum == 0) or is_last_batch

                if should_step:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    # global_step counts optimizer steps (not batches) so that
                    # logging_steps / save_steps / eval_steps behave consistently
                    # regardless of gradient_accumulation_steps.
                    self.global_step += 1

                # Extract confidence and correctness from the training outputs
                with torch.no_grad():
                    confidences = self.compute_confidence(outputs, batch)
                    corrects = self.compute_correctness(outputs, batch)
                    if loss_raw.dim() == 1 and loss_raw.numel() == batch_size:
                        batch_losses = loss_raw.detach()
                    else:
                        batch_losses = torch.full(
                            (batch_size,), loss_for_backward.item(), device=self.device
                        )
                self._update_sample_states(sample_ids, batch_losses, confidences, corrects)

                # Update tracking (use unscaled loss for metrics)
                epoch_loss += loss_for_backward.item() * batch_size
                epoch_samples += batch_size
                total_loss += loss_for_backward.item() * batch_size
                samples_trained += batch_size

                # Advance curriculum and adjust LR on phase change
                if self.curriculum:
                    self.curriculum.step(batch_size)
                    if self.curriculum.phase_changed:
                        new_lr = self.curriculum.adjust_learning_rate(self.args.learning_rate)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                        # Sync scheduler's base LRs so warmup math stays consistent
                        scheduler.base_lrs = [new_lr] * len(scheduler.base_lrs)
                        print(
                            f"\n[GEKO] Phase → {self.curriculum.current_phase.value}, "
                            f"LR → {new_lr:.2e}"
                        )

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_for_backward.item():.4f}",
                    'phase': self.curriculum.current_phase.value if self.curriculum else 'N/A',
                })

                # Logging
                if self.global_step % self.args.logging_steps == 0 and epoch_samples > 0:
                    avg_loss = epoch_loss / epoch_samples
                    print(f"\n[Step {self.global_step}] Loss: {avg_loss:.4f}")

                # Eval
                if self.eval_dataset is not None and self.global_step % self.args.eval_steps == 0:
                    self._run_eval()

                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()

            # End of epoch
            if epoch_samples > 0:
                avg_epoch_loss = epoch_loss / epoch_samples
                print(f"\n[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")
            else:
                print(
                    f"\n[Epoch {epoch+1}] No batches in this epoch "
                    "(empty dataset or all samples skipped)."
                )

        # Final save (skip if user manages checkpointing themselves)
        if self.args.save_at_end:
            self.save_checkpoint()

        return {
            'total_loss': total_loss / samples_trained if samples_trained > 0 else 0.0,
            'samples_trained': samples_trained,
            'efficiency': self.get_efficiency_report(),
        }

    def _run_eval(self) -> Optional[float]:
        """
        Run evaluation on eval_dataset and return average loss.

        Called automatically every eval_steps batches during training if
        eval_dataset was provided to GEKOTrainer. Returns None if no eval_dataset.
        """
        if self.eval_dataset is None:
            return None

        self.model.eval()
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.device.type == 'cuda'),
        )
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                batch.pop('sample_id', None)  # eval dataset may not have sample_id
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                if loss.dim() > 0:
                    loss = loss.mean()
                bs = self._get_batch_size(batch)
                total_loss += loss.item() * bs
                total_samples += bs

        self.model.train()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"\n[Eval @ step {self.global_step}] Eval Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model and GEKO state."""
        path = path or os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(path, exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

        # Save GEKO state using to_dict() for clean JSON serialization
        geko_state = {
            'sample_states': {k: v.to_dict() for k, v in self.sample_states.items()},
            'partition_history': [p.to_dict() for p in self.partition_history],
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }
        with open(os.path.join(path, "geko_state.json"), 'w') as f:
            json.dump(geko_state, f, indent=2)

        print(f"[GEKO] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load model and GEKO state from a checkpoint directory.

        Restores global_step, current_epoch, and all per-sample states so
        training can resume exactly where it left off.

        Args:
            path: Path to the checkpoint directory saved by save_checkpoint()

        Note:
            For HuggingFace models (save_pretrained / from_pretrained), the model
            weights must be loaded separately before calling this method:
                model = AutoModel.from_pretrained(path)
                trainer = GEKOTrainer(model=model, ...)
                trainer.load_checkpoint(path)  # restores GEKO state only
        """
        # Load plain PyTorch model weights if present
        model_pt = os.path.join(path, "model.pt")
        if os.path.exists(model_pt):
            self.model.load_state_dict(
                torch.load(model_pt, map_location=self.device, weights_only=True)
            )

        # Load GEKO state
        state_path = os.path.join(path, "geko_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"No geko_state.json found in '{path}'. "
                f"Make sure '{path}' is a directory created by save_checkpoint()."
            )

        with open(state_path) as f:
            geko_state = json.load(f)

        self.global_step = geko_state['global_step']
        self.current_epoch = geko_state['current_epoch']

        # Restore SampleState objects from serialised dicts
        for sample_id, d in geko_state['sample_states'].items():
            if sample_id not in self.sample_states:
                continue
            s = self.sample_states[sample_id]
            s.bucket = Bucket(d['bucket'])
            s.q_value = d['q_value']
            s.confidence = d['confidence']
            s.quality = d['quality']
            s.loss_history = deque(d['loss_history'], maxlen=5)
            s.times_seen = d['times_seen']
            s.last_loss = d['last_loss']
            s.frozen_at_epoch = d['frozen_at_epoch']
            s.correct = d['correct']
            s.consecutive_frozen_epochs = d.get('consecutive_frozen_epochs', 0)

        # Restore partition history (new format: list of dicts; old format: list of strings)
        ph = geko_state.get('partition_history', [])
        if ph and isinstance(ph[0], dict):
            self.partition_history = [PartitionStats.from_dict(d) for d in ph]
        else:
            self.partition_history = []
            if ph:
                print(
                    "[GEKO] Checkpoint was saved with an older format; "
                    "efficiency history was not restored. New partitions will be recorded from this run."
                )

        print(
            f"[GEKO] Resumed from '{path}' "
            f"(step={self.global_step}, epoch={self.current_epoch})"
        )

    def get_efficiency_report(self) -> Dict:
        """
        Get GEKO efficiency report.

        Shows how much compute was saved by skipping mastered samples.
        """
        if not self.partition_history:
            return {}

        total_samples = len(self.train_dataset)
        if total_samples == 0:
            return {}

        latest = self.partition_history[-1]
        initial = self.partition_history[0] if len(self.partition_history) > 1 else latest

        # Compute savings
        samples_skipped = latest.freeze_count
        compute_saved = samples_skipped / total_samples

        return {
            'total_samples': total_samples,
            'samples_mastered': latest.freeze_count,
            'samples_skipped': samples_skipped,
            'compute_saved_percent': f"{compute_saved:.1%}",
            'final_accuracy': f"{latest.accuracy:.1%}",
            'bucket_distribution': str(latest),
            'improvement': {
                'freeze_change': latest.freeze_ratio - initial.freeze_ratio,
                'hard_change': latest.hard_ratio - initial.hard_ratio,
            }
        }

    def get_hard_samples(self) -> List[str]:
        """Get IDs of current HARD samples (confident-wrong)."""
        return self.partitioner.get_bucket_samples(self.sample_states, Bucket.HARD)

    def get_frozen_samples(self) -> List[str]:
        """Get IDs of FROZEN samples (mastered)."""
        return self.partitioner.get_bucket_samples(self.sample_states, Bucket.FREEZE)
