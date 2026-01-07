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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
import json
import os
from tqdm import tqdm

from .core import Bucket, SampleState, GEKOConfig
from .partitioner import SamplePartitioner, PartitionStats
from .curriculum import MountainCurriculum, CurriculumPhase


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
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    seed: int = 42


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
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[GEKOConfig] = None,
        args: Optional[GEKOTrainingArgs] = None,
        compute_confidence: Optional[Callable] = None,
        compute_correctness: Optional[Callable] = None,
    ):
        """
        Initialize GEKO Trainer.

        Args:
            model: Any PyTorch model (HuggingFace, custom, etc.)
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            config: GEKO configuration
            args: Training arguments
            compute_confidence: Custom function to compute model confidence
            compute_correctness: Custom function to check if prediction is correct
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or GEKOConfig()
        self.args = args or GEKOTrainingArgs()

        # Custom evaluation functions
        self.compute_confidence = compute_confidence or self._default_confidence
        self.compute_correctness = compute_correctness or self._default_correctness

        # GEKO components
        self.partitioner = SamplePartitioner(self.config)
        self.curriculum = MountainCurriculum(
            total_samples=len(train_dataset) * self.args.num_epochs,
            config=self.config
        ) if self.config.use_curriculum else None

        # Sample states (tracks learning progress per sample)
        self.sample_states: Dict[str, SampleState] = {}
        self._init_sample_states()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.partition_history: List[PartitionStats] = []

        # Device
        self.device = next(model.parameters()).device

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
        model: nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Default confidence computation using softmax probability.

        Override this for custom confidence metrics.
        """
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Get probability of predicted token
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values

            # Average confidence across sequence
            confidence = max_probs.mean(dim=-1)

        return confidence

    def _default_correctness(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Default correctness check using loss threshold.

        Override this for task-specific correctness.
        """
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Sample is "correct" if loss is below threshold
            if loss.dim() == 0:
                # Single loss value
                correct = (loss < threshold).unsqueeze(0)
            else:
                correct = loss < threshold

        return correct

    def _get_sample_weights(self) -> List[float]:
        """Get sampling weights based on bucket assignments."""
        weights = []
        for idx in range(len(self.train_dataset)):
            sample_id = str(idx)
            state = self.sample_states[sample_id]
            weight = self.config.get_bucket_weight(state.bucket)

            # Apply curriculum adjustment
            if self.curriculum:
                curr_weights = self.curriculum.get_current_weights()
                bucket_idx = {Bucket.HARD: 0, Bucket.FOCUS: 1, Bucket.LIGHT: 2}.get(state.bucket, -1)
                if bucket_idx >= 0:
                    weight = curr_weights[bucket_idx]

            weights.append(max(weight, 0.001))  # Minimum weight to avoid zero

        return weights

    def _create_dataloader(self, weighted: bool = True) -> DataLoader:
        """Create dataloader with GEKO-weighted sampling."""
        if weighted:
            weights = self._get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

    def _update_sample_states(
        self,
        sample_ids: List[str],
        losses: torch.Tensor,
        confidences: torch.Tensor,
        corrects: torch.Tensor
    ):
        """Update sample states after a training step."""
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.sample_states:
                self.sample_states[sample_id].update(
                    loss=losses[i].item() if losses.dim() > 0 else losses.item(),
                    confidence=confidences[i].item() if confidences.dim() > 0 else confidences.item(),
                    correct=corrects[i].item() if corrects.dim() > 0 else bool(corrects),
                    epoch=self.current_epoch,
                )

    def partition_samples(self) -> PartitionStats:
        """
        Re-partition all samples based on current model performance.

        Call this at the start of each epoch to update bucket assignments.
        """
        stats = self.partitioner.partition(self.sample_states, self.current_epoch)
        self.partition_history.append(stats)

        if self.config.log_bucket_stats:
            print(f"\n[GEKO] Epoch {self.current_epoch} Partition: {stats}")

        return stats

    def train(self):
        """
        Main training loop with GEKO optimization.

        Returns:
            Dict with training results and efficiency metrics
        """
        self.model.train()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.args.fp16 else None

        # Training loop
        total_loss = 0
        samples_trained = 0

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch

            # Re-partition at start of each epoch
            if epoch % self.config.repartition_every == 0:
                stats = self.partition_samples()

                # Check for early stopping
                if self.partitioner.should_stop_early(stats):
                    print(f"\n[GEKO] Early stopping: {stats.freeze_ratio:.1%} samples mastered!")
                    break

            # Create weighted dataloader
            dataloader = self._create_dataloader(weighted=True)

            # Epoch training
            epoch_loss = 0
            epoch_samples = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get sample IDs (if available in batch)
                sample_ids = batch.pop('sample_id', [str(i) for i in range(len(batch['input_ids']))])

                # Forward pass
                optimizer.zero_grad()

                if self.args.fp16 and scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                # Handle DataParallel
                if loss.dim() > 0:
                    loss = loss.mean()

                # Backward pass
                if self.args.fp16 and scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                # Update tracking
                batch_size = len(batch['input_ids'])
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                total_loss += loss.item() * batch_size
                samples_trained += batch_size
                self.global_step += 1

                # Update curriculum
                if self.curriculum:
                    self.curriculum.step(batch_size)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'phase': self.curriculum.current_phase.value if self.curriculum else 'N/A',
                })

                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = epoch_loss / epoch_samples
                    print(f"\n[Step {self.global_step}] Loss: {avg_loss:.4f}")

                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_samples
            print(f"\n[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")

        # Final save
        self.save_checkpoint()

        return {
            'total_loss': total_loss / samples_trained,
            'samples_trained': samples_trained,
            'efficiency': self.get_efficiency_report(),
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model and GEKO state."""
        path = path or os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(path, exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

        # Save GEKO state
        geko_state = {
            'sample_states': {k: vars(v) for k, v in self.sample_states.items()},
            'partition_history': [str(p) for p in self.partition_history],
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
        }
        with open(os.path.join(path, "geko_state.json"), 'w') as f:
            json.dump(geko_state, f, indent=2, default=str)

        print(f"[GEKO] Checkpoint saved to {path}")

    def get_efficiency_report(self) -> Dict:
        """
        Get GEKO efficiency report.

        Shows how much compute was saved by skipping mastered samples.
        """
        if not self.partition_history:
            return {}

        latest = self.partition_history[-1]
        initial = self.partition_history[0] if len(self.partition_history) > 1 else latest

        # Compute savings
        total_samples = len(self.train_dataset)
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
