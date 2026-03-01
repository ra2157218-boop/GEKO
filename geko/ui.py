"""
GEKO Rich Terminal UI

Provides beautiful Rich-based training output with colored panels,
progress bars, and bucket distribution charts.

Falls back to plain-text output automatically if:
  - rich is not installed
  - use_rich_ui=False in GEKOTrainingArgs
"""

from __future__ import annotations
from typing import Any, Optional

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        SpinnerColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class GEKORichUI:
    """
    UI layer for GEKOTrainer.

    Provides Rich-based terminal output when rich is installed,
    gracefully falls back to plain print() otherwise.

    Usage:
        ui = GEKORichUI(use_rich=True)
        ui.print_training_header(args, config, n_train=1000, n_eval=100,
                                 device="cuda", n_workers=4)
    """

    def __init__(self, use_rich: bool = True):
        self.enabled = use_rich and RICH_AVAILABLE
        if use_rich and not RICH_AVAILABLE:
            print(
                "[GEKO] rich not installed — using plain output. "
                "Install with: pip install rich  or  pip install gekolib[rich]"
            )
        self.console = Console() if self.enabled else None

    # ── Training Header ──────────────────────────────────────────────────────

    def print_training_header(
        self,
        args: Any,
        config: Any,
        n_train: int,
        n_eval: Optional[int],
        device: str,
        n_workers: int,
    ):
        """Print the training configuration header."""
        use_bf16 = args.bf16
        use_fp16 = args.fp16 and not use_bf16
        precision_str = "BF16" if use_bf16 else ("FP16" if use_fp16 else "FP32")

        if not self.enabled:
            print(
                f"\n{'='*55}\n"
                f"  GEKO Training\n"
                f"{'='*55}\n"
                f"  Samples           : {n_train}\n"
                f"  Epochs            : {args.num_epochs}\n"
                f"  Batch size        : {args.batch_size}\n"
                f"  Device            : {device}\n"
                f"  Precision         : {precision_str}\n"
                f"  Grad accum        : {args.gradient_accumulation_steps}\n"
                f"  Grad checkpointing: {'ON' if args.gradient_checkpointing else 'OFF'}\n"
                f"  torch.compile     : {'ON' if args.compile_model else 'OFF'}\n"
                f"  8-bit optimizer   : {'ON' if args.use_8bit_optimizer else 'OFF'}\n"
                f"  DataLoader workers: {n_workers}\n"
                f"  Warmup steps      : {args.warmup_steps}\n"
                f"  Curriculum        : {'ON' if config.use_curriculum else 'OFF'}\n"
                f"  Config            :\n{config}\n"
                f"{'='*55}\n"
            )
            return

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        eval_suffix = f"  (eval: {n_eval:,})" if n_eval else ""
        rows = [
            ("Samples",            f"{n_train:,}{eval_suffix}"),
            ("Epochs",             str(args.num_epochs)),
            ("Batch size",         str(args.batch_size)),
            ("Device",             device),
            ("Precision",          precision_str),
            ("Grad accum",         str(args.gradient_accumulation_steps)),
            ("Grad checkpointing", "ON" if args.gradient_checkpointing else "OFF"),
            ("torch.compile",      "ON" if args.compile_model else "OFF"),
            ("8-bit optimizer",    "ON" if args.use_8bit_optimizer else "OFF"),
            ("DataLoader workers", str(n_workers)),
            ("Warmup steps",       str(args.warmup_steps)),
            ("Curriculum",         "ON" if config.use_curriculum else "OFF"),
            ("Freeze threshold",
             f"{config.freeze_confidence:.0%} conf / {config.freeze_quality:.0%} quality"),
        ]
        for key, val in rows:
            table.add_row(key, val)

        self.console.print(
            Panel(table, title="[bold green]GEKO Training[/bold green]", border_style="green")
        )

    # ── Epoch Progress Bar ───────────────────────────────────────────────────

    def start_epoch_progress(self, epoch: int, total_epochs: int, n_batches: int) -> Any:
        """
        Start a progress bar for one epoch.
        Returns a handle passed to update_progress() and finish_epoch_progress().
        """
        if self.enabled:
            progress = Progress(
                SpinnerColumn(),
                TextColumn(
                    f"[bold cyan]Epoch {epoch}/{total_epochs}[/bold cyan]",
                    justify="right",
                ),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("[yellow]loss={task.fields[loss]:.4f}[/yellow]"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            task_id = progress.add_task("training", total=n_batches, loss=0.0)
            progress.start()
            return (progress, task_id)
        else:
            from tqdm import tqdm
            return tqdm(range(n_batches), desc=f"Epoch {epoch}/{total_epochs}")

    def update_progress(self, pbar: Any, loss: float):
        """Advance the progress bar by 1 and update the displayed loss."""
        if self.enabled and isinstance(pbar, tuple):
            progress, task_id = pbar
            progress.update(task_id, advance=1, loss=loss)
        else:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")

    def finish_epoch_progress(self, pbar: Any):
        """Stop and close the progress bar after an epoch completes."""
        if self.enabled and isinstance(pbar, tuple):
            progress, _ = pbar
            progress.stop()
        else:
            pbar.close()

    # ── Partition Panel ──────────────────────────────────────────────────────

    def print_partition(self, epoch: int, stats: Any):
        """Print the bucket distribution after sample partitioning."""
        if not self.enabled:
            print(f"\n[GEKO] Epoch {epoch} Partition: {stats}")
            return

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Icon",   no_wrap=True, width=2)
        table.add_column("Bucket", style="bold",  no_wrap=True, width=8)
        table.add_column("Bar",    no_wrap=True, width=22)
        table.add_column("Pct",    no_wrap=True, width=6)
        table.add_column("Count",  style="dim",   no_wrap=True)

        buckets = [
            ("🔵", "FREEZE", stats.freeze_ratio, stats.freeze_count, "blue"),
            ("🟢", "LIGHT",  stats.light_ratio,  stats.light_count,  "green"),
            ("🟠", "FOCUS",  stats.focus_ratio,  stats.focus_count,  "yellow"),
            ("🔴", "HARD",   stats.hard_ratio,   stats.hard_count,   "red"),
        ]
        for icon, name, ratio, count, color in buckets:
            bar = self._bar_str(ratio, width=20)
            table.add_row(
                icon,
                f"[{color}]{name}[/{color}]",
                f"[{color}]{bar}[/{color}]",
                f"{ratio:.1%}",
                f"({count})",
            )

        saved_line = Text(f"\nCompute saved this epoch: ~{stats.freeze_ratio:.0%}", style="bold green")
        content = Group(table, saved_line)
        self.console.print(Panel(
            content,
            title=f"[bold]Epoch {epoch} — Sample Distribution[/bold]",
            border_style="cyan",
        ))

    # ── Log Methods ──────────────────────────────────────────────────────────

    def log_step_loss(self, step: int, loss: float):
        if self.enabled:
            self.console.log(f"[dim][Step {step}] Loss: {loss:.4f}[/dim]")
        else:
            print(f"\n[Step {step}] Loss: {loss:.4f}")

    def log_eval_loss(self, step: int, loss: float):
        if self.enabled:
            self.console.log(f"[bold cyan][Eval @ step {step}] Eval Loss: {loss:.4f}[/bold cyan]")
        else:
            print(f"\n[Eval @ step {step}] Eval Loss: {loss:.4f}")

    def log_epoch_loss(self, epoch: int, loss: float):
        if self.enabled:
            self.console.log(f"[bold][Epoch {epoch}] Avg Loss: {loss:.4f}[/bold]")
        else:
            print(f"\n[Epoch {epoch}] Average Loss: {loss:.4f}")

    def log_phase_change(self, phase: str, lr: float):
        if self.enabled:
            self.console.log(f"[bold magenta]Phase → {phase}  LR → {lr:.2e}[/bold magenta]")
        else:
            print(f"\n[GEKO] Phase → {phase}, LR → {lr:.2e}")

    def log_checkpoint(self, path: str):
        if self.enabled:
            self.console.log(f"[green]Checkpoint saved → {path}[/green]")
        else:
            print(f"[GEKO] Checkpoint saved to {path}")

    def log_message(self, msg: str):
        if self.enabled:
            self.console.log(msg)
        else:
            print(msg)

    def log_warning(self, msg: str):
        if self.enabled:
            self.console.log(f"[yellow]⚠ {msg}[/yellow]")
        else:
            print(f"[GEKO] Warning: {msg}")

    def log_early_stop(self, freeze_ratio: float):
        if self.enabled:
            self.console.print(Panel(
                f"[bold green]{freeze_ratio:.1%} of samples mastered — training complete![/bold green]",
                title="[bold green]Early Stopping[/bold green]",
                border_style="green",
            ))
        else:
            print(f"\n[GEKO] Early stopping: {freeze_ratio:.1%} samples mastered!")

    def log_pruned(self, n: int, threshold: int, active: int):
        if self.enabled:
            self.console.log(
                f"[dim]Pruned {n} samples (frozen for {threshold}+ epochs). "
                f"Active dataset: {active:,} samples.[/dim]"
            )
        else:
            print(
                f"[GEKO] Pruned {n} samples "
                f"(frozen for {threshold}+ epochs). "
                f"Active dataset: {active} samples."
            )

    def log_dataloader_reuse(self):
        if self.enabled:
            self.console.log("[dim]Bucket distribution stable — reusing dataloader[/dim]")
        else:
            print("[GEKO] Bucket distribution stable — reusing dataloader")

    def log_batch_mode_warning(self):
        msg = (
            "Using batch-level correctness (model returns scalar loss). "
            "For per-sample bucketing, override compute_correctness; see API Reference."
        )
        if self.enabled:
            self.console.log(f"[yellow]⚠ {msg}[/yellow]")
        else:
            print(f"[GEKO] {msg}")

    def log_all_mastered(self):
        msg = (
            "All samples are mastered or at max_times_seen; "
            "skipping this epoch. GEKO will stop after the next partition check."
        )
        if self.enabled:
            self.console.log(f"[yellow]⚠ {msg}[/yellow]")
        else:
            print(f"\n[GEKO] {msg}")

    def log_no_batches(self, epoch: int):
        if self.enabled:
            self.console.log(
                f"[yellow]⚠ Epoch {epoch}: no batches "
                f"(empty dataset or all samples skipped).[/yellow]"
            )
        else:
            print(
                f"\n[Epoch {epoch}] No batches in this epoch "
                "(empty dataset or all samples skipped)."
            )

    def log_resumed(self, path: str, step: int, epoch: int):
        if self.enabled:
            self.console.log(
                f"[green]Resumed from '{path}' (step={step}, epoch={epoch})[/green]"
            )
        else:
            print(f"[GEKO] Resumed from '{path}' (step={step}, epoch={epoch})")

    def log_old_checkpoint(self):
        msg = (
            "Checkpoint was saved with an older format; "
            "efficiency history was not restored. "
            "New partitions will be recorded from this run."
        )
        if self.enabled:
            self.console.log(f"[yellow]⚠ {msg}[/yellow]")
        else:
            print(f"[GEKO] {msg}")

    # ── Training Summary ─────────────────────────────────────────────────────

    def print_summary(self, result: dict, report: dict, pruned: int):
        """Print the final training summary panel (Rich only)."""
        if not self.enabled:
            return  # plain-text summary is handled by the caller

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Key",   style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Final avg loss",   f"{result.get('total_loss', 0):.4f}")
        table.add_row("Samples trained",  f"{result.get('samples_trained', 0):,}")
        if report:
            table.add_row("Samples mastered", f"{report.get('samples_mastered', 0):,}")
            table.add_row("Samples pruned",   f"{pruned:,}")
            table.add_row("Compute saved",    report.get('compute_saved_percent', 'N/A'))
            table.add_row("Final accuracy",   report.get('final_accuracy', 'N/A'))

        self.console.print(
            Panel(table, title="[bold green]Training Complete[/bold green]", border_style="green")
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bar_str(ratio: float, width: int = 20) -> str:
        """Generate a text progress bar like '████░░░░░░'."""
        filled = round(ratio * width)
        return "█" * filled + "░" * (width - filled)
