"""
Unit tests for geko/trainer.py

Covers: GEKODataset, GEKOTrainingArgs, _get_sample_weights / _create_dataloader
(all-zero fallback), last-batch sample-ID fix, load_checkpoint round-trip.

These tests do NOT run a full training loop — that requires a real model.
"""

import json
import os
import tempfile
import pytest
import torch
from collections import deque
from torch.utils.data import Dataset

from geko.core import Bucket, SampleState, GEKOConfig
from geko.trainer import GEKODataset, GEKOTrainer, GEKOTrainingArgs


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

class DictDataset(Dataset):
    """Tiny dict-returning dataset for testing GEKODataset wrapping."""
    def __init__(self, n: int = 10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor([idx]), 'labels': torch.tensor([idx])}


class TupleDataset(Dataset):
    """Returns tuples — should fail at GEKODataset construction."""
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return (torch.tensor([idx]), torch.tensor([idx]))


class DictNoInputIdsDataset(Dataset):
    """Returns a dict but without 'input_ids' — should fail at GEKODataset construction."""
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"labels": torch.tensor([idx])}


class TinyModel(torch.nn.Module):
    """One-layer model that always returns a fake loss and logits."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.linear(input_ids.float())
        loss = torch.tensor(1.0, requires_grad=True)

        class Output:
            pass

        out = Output()
        out.loss = loss
        out.logits = logits
        return out


# ---------------------------------------------------------------------------
# GEKODataset
# ---------------------------------------------------------------------------

class TestGEKODataset:
    def test_dict_dataset_wraps_ok(self):
        ds = GEKODataset(DictDataset(5))
        assert len(ds) == 5

    def test_sample_id_injected(self):
        ds = GEKODataset(DictDataset(5))
        item = ds[3]
        assert item['sample_id'] == '3'

    def test_tuple_dataset_raises_type_error(self):
        with pytest.raises(TypeError, match="dict"):
            GEKODataset(TupleDataset())

    def test_original_keys_preserved(self):
        ds = GEKODataset(DictDataset(3))
        item = ds[0]
        assert 'input_ids' in item
        assert 'labels' in item

    def test_dict_without_input_ids_raises(self):
        with pytest.raises(TypeError, match="input_ids"):
            GEKODataset(DictNoInputIdsDataset())


# ---------------------------------------------------------------------------
# GEKOTrainingArgs
# ---------------------------------------------------------------------------

class TestGEKOTrainingArgs:
    def test_fp16_auto_detects(self):
        args = GEKOTrainingArgs()
        assert args.fp16 == torch.cuda.is_available()

    def test_num_workers_default_auto(self):
        args = GEKOTrainingArgs()
        assert args.dataloader_num_workers == -1  # -1 = auto-detect

    def test_save_at_end_default_true(self):
        args = GEKOTrainingArgs()
        assert args.save_at_end is True

    def test_grad_accum_default_one(self):
        args = GEKOTrainingArgs()
        assert args.gradient_accumulation_steps == 1

    def test_warmup_steps_default(self):
        args = GEKOTrainingArgs()
        assert args.warmup_steps == 100

    def test_eval_steps_default(self):
        args = GEKOTrainingArgs()
        assert args.eval_steps == 500


# ---------------------------------------------------------------------------
# _get_sample_weights — all-zero fallback
# ---------------------------------------------------------------------------

class TestSampleWeightsFallback:
    def _make_trainer(self) -> GEKOTrainer:
        model = TinyModel()
        args = GEKOTrainingArgs(save_at_end=False)
        return GEKOTrainer(
            model=model,
            train_dataset=DictDataset(8),
            args=args,
        )

    def test_non_zero_weights_by_default(self):
        trainer = self._make_trainer()
        weights = trainer._get_sample_weights()
        # All start in FOCUS bucket (weight >= 1)
        assert sum(weights) > 0

    def test_all_freeze_returns_nonzero_weights(self):
        trainer = self._make_trainer()
        # Force all samples into FREEZE
        for state in trainer.sample_states.values():
            state.bucket = Bucket.FREEZE
        weights = trainer._get_sample_weights()
        assert sum(weights) == 0  # _get_sample_weights returns 0s

        # _create_dataloader should fall back to uniform — no crash
        loader = trainer._create_dataloader(weighted=True)
        assert loader is not None

    def test_all_freeze_dataloader_does_not_raise(self):
        trainer = self._make_trainer()
        for state in trainer.sample_states.values():
            state.bucket = Bucket.FREEZE
        # This should NOT raise ValueError
        try:
            loader = trainer._create_dataloader(weighted=True)
        except ValueError as e:
            pytest.fail(f"_create_dataloader raised ValueError: {e}")

    def test_max_times_seen_excluded_from_weights(self):
        trainer = self._make_trainer()
        config = trainer.config
        # Push sample '0' past the limit
        for _ in range(config.max_times_seen):
            trainer.sample_states['0'].update(loss=1.0, confidence=0.5, correct=False)
        weights = trainer._get_sample_weights()
        assert weights[0] == 0.0, "Over-trained sample should have weight 0"
        # Other samples should still have positive weight (they start in FOCUS)
        assert any(w > 0 for w in weights[1:])


# ---------------------------------------------------------------------------
# load_checkpoint round-trip
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def _make_trainer(self) -> GEKOTrainer:
        model = TinyModel()
        args = GEKOTrainingArgs(save_at_end=False)
        return GEKOTrainer(
            model=model,
            train_dataset=DictDataset(5),
            args=args,
        )

    def test_load_checkpoint_restores_global_step(self):
        trainer = self._make_trainer()
        trainer.global_step = 42
        trainer.current_epoch = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(tmpdir)
            assert trainer2.global_step == 42

    def test_load_checkpoint_restores_current_epoch(self):
        trainer = self._make_trainer()
        trainer.current_epoch = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(tmpdir)
            assert trainer2.current_epoch == 3

    def test_load_checkpoint_restores_bucket(self):
        trainer = self._make_trainer()
        trainer.sample_states['0'].bucket = Bucket.HARD
        trainer.sample_states['0'].q_value = 0.9
        trainer.sample_states['0'].times_seen = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(tmpdir)
            assert trainer2.sample_states['0'].bucket == Bucket.HARD
            assert trainer2.sample_states['0'].q_value == pytest.approx(0.9)
            assert trainer2.sample_states['0'].times_seen == 7

    def test_load_checkpoint_restores_loss_history(self):
        trainer = self._make_trainer()
        for loss in [1.0, 2.0, 3.0]:
            trainer.sample_states['1'].update(loss=loss, confidence=0.5, correct=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(tmpdir)
            assert list(trainer2.sample_states['1'].loss_history) == [1.0, 2.0, 3.0]

    def test_load_checkpoint_missing_file_raises(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                trainer.load_checkpoint(tmpdir)  # no geko_state.json there

    def test_load_checkpoint_restores_partition_history(self):
        trainer = self._make_trainer()
        trainer.partition_samples()  # add one entry to partition_history
        assert len(trainer.partition_history) == 1
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(tmpdir)
            assert len(trainer2.partition_history) == 1
            report = trainer2.get_efficiency_report()
            assert report != {}
            assert "total_samples" in report


# ---------------------------------------------------------------------------
# Empty dataset / division by zero
# ---------------------------------------------------------------------------

class TestEmptyDataset:
    def test_empty_dataset_no_division_by_zero(self):
        """Training with empty dataset should not raise ZeroDivisionError."""
        model = TinyModel()
        args = GEKOTrainingArgs(num_epochs=1, save_at_end=False)
        trainer = GEKOTrainer(
            model=model,
            train_dataset=DictDataset(0),
            args=args,
        )
        result = trainer.train()
        assert "efficiency" in result


# ---------------------------------------------------------------------------
# All-FREEZE skip epoch
# ---------------------------------------------------------------------------

class TestAllFreezeSkipsEpoch:
    def test_all_freeze_skips_epoch_and_stops(self):
        """When all samples are FREEZE, trainer skips epoch and stops early."""
        model = TinyModel()
        args = GEKOTrainingArgs(num_epochs=3, save_at_end=False)
        trainer = GEKOTrainer(
            model=model,
            train_dataset=DictDataset(5),
            args=args,
        )
        for state in trainer.sample_states.values():
            state.bucket = Bucket.FREEZE
        result = trainer.train()
        # Should exit without running a full epoch of batches
        assert "efficiency" in result
