"""
GEKO Web App

A user-friendly Gradio interface for fine-tuning any HuggingFace model with GEKO.
No coding required — pick a model, pick a dataset, click Start.

Usage:
    python -m geko.app
    # or after pip install gekolib[app]:
    geko-app

Features:
- Model & dataset picker (any HuggingFace name)
- Optional eval dataset support
- Training config sliders (epochs, LR, batch size, precision)
- GEKO config panel (presets, freeze/focus/quality thresholds, curriculum, pruning)
- LoRA / PEFT integration
- Efficiency options (torch.compile, gradient checkpointing, 8-bit optimizer)
- Live loss curve (Plotly, updates every batch)
- Live bucket distribution chart (FREEZE / LIGHT / FOCUS / HARD)
- Steps/sec + ETA tracking
- Eval loss display
- Training summary card
- Resume from checkpoint
- Stop button that safely halts training mid-epoch
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
import traceback
from collections import deque
from typing import Any, List, Optional

import torch

# ── Optional deps ──────────────────────────────────────────────────────────────
# Pre-load pandas at module level so plotly's validator never hits a partially
# initialised module when transformers/datasets import it in a background thread.
try:
    import pandas as _pd
except ImportError:
    _pd = None

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from geko import GEKOTrainer, GEKOConfig, GEKOTrainingArgs
from geko.ui import GEKORichUI

# ── Global state ───────────────────────────────────────────────────────────────
_stop_event: Optional[threading.Event] = None
_current_thread: Optional[threading.Thread] = None


# ══════════════════════════════════════════════════════════════════════════════
# Queue-based UI  (routes all trainer output to the browser)
# ══════════════════════════════════════════════════════════════════════════════

class _GradioUI(GEKORichUI):
    """
    GEKORichUI subclass that feeds events into a queue instead of printing.
    The Gradio generator polls this queue and yields component updates.

    Raises StopIteration inside update_progress() when the stop button is clicked,
    which propagates out of the training loop cleanly.
    """

    def __init__(self, q: queue.Queue, stop_event: threading.Event):
        super().__init__(use_rich=False)   # disable Rich output in the app
        self._q = q
        self._stop = stop_event
        self._last_logged_step = -1       # dedup: prevent step_loss spam per optimizer step
        self._last_checkpoint_path = None  # dedup: prevent repeated checkpoint-0 saves

    # ── Progress (called every batch — also the stop check point) ────────────

    def update_progress(self, pbar: Any, loss: float):
        if self._stop.is_set():
            raise StopIteration("Training stopped by user")
        self._q.put({'type': 'step', 'loss': loss})

    def start_epoch_progress(self, epoch, total_epochs, n_batches):
        return None   # no tqdm or rich bar; progress shown in Gradio

    def finish_epoch_progress(self, pbar):
        pass

    # ── Partition stats (called once per epoch) ───────────────────────────────

    def print_partition(self, epoch: int, stats: Any):
        self._q.put({
            'type':   'partition',
            'epoch':  epoch,
            'freeze': stats.freeze_count,
            'light':  stats.light_count,
            'focus':  stats.focus_count,
            'hard':   stats.hard_count,
            'total':  stats.total,
        })

    # ── Step / epoch logs ─────────────────────────────────────────────────────

    def log_step_loss(self, step: int, loss: float):
        # Only emit once per unique optimizer step — trainer checks fire every batch
        if step == self._last_logged_step:
            return
        self._last_logged_step = step
        self._q.put({'type': 'step_loss', 'step': step, 'loss': loss})

    def log_epoch_loss(self, epoch: int, loss: float):
        self._q.put({'type': 'epoch_loss', 'epoch': epoch, 'loss': loss})
        self._q.put({'type': 'log', 'msg': f'[Epoch {epoch}] Avg Loss: {loss:.4f}'})

    def log_eval_loss(self, step: int, loss: float):
        # Dedicated event type so the UI can display it in the Eval Loss stat box
        self._q.put({'type': 'eval_loss', 'step': step, 'loss': loss})

    def log_phase_change(self, phase: str, lr: float):
        self._q.put({'type': 'log', 'msg': f'Phase → {phase}  LR → {lr:.2e}'})

    def log_checkpoint(self, path: str):
        # Only log once per unique path — trainer save check fires every batch at step 0
        if path == self._last_checkpoint_path:
            return
        self._last_checkpoint_path = path
        self._q.put({'type': 'log', 'msg': f'✓ Checkpoint saved → {path}'})

    def log_early_stop(self, freeze_ratio: float):
        self._q.put({'type': 'log', 'msg': f'🏁 Early stop: {freeze_ratio:.1%} mastered'})

    def log_message(self, msg: str):
        self._q.put({'type': 'log', 'msg': msg})

    def log_warning(self, msg: str):
        self._q.put({'type': 'log', 'msg': f'⚠ {msg}'})

    def log_pruned(self, n: int, threshold: int, active: int):
        self._q.put({'type': 'log', 'msg': f'Pruned {n} samples. Active: {active:,}'})

    def log_dataloader_reuse(self):
        self._q.put({'type': 'log', 'msg': 'Bucket stable — reusing dataloader'})

    def log_batch_mode_warning(self):
        self._q.put({'type': 'log', 'msg': '⚠ Batch-level correctness (scalar loss). For per-sample bucketing, override compute_correctness.'})

    def log_all_mastered(self):
        self._q.put({'type': 'log', 'msg': '⚠ All samples mastered — skipping epoch'})

    def log_no_batches(self, epoch: int):
        self._q.put({'type': 'log', 'msg': f'⚠ Epoch {epoch}: no batches'})

    def log_resumed(self, path: str, step: int, epoch: int):
        self._q.put({'type': 'log', 'msg': f'Resumed from {path} (step={step})'})

    def log_old_checkpoint(self):
        self._q.put({'type': 'log', 'msg': '⚠ Old checkpoint format — efficiency history not restored'})

    # ── Header & summary (suppressed in GUI — shown via status/charts) ────────

    def print_training_header(self, *args, **kwargs):
        pass   # config is visible in the GUI panels

    def print_summary(self, result: dict, report: dict, pruned: int):
        self._q.put({'type': 'done', 'result': result, 'report': report, 'pruned': pruned})


# ══════════════════════════════════════════════════════════════════════════════
# Dataset builder  (auto-detects column format)
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleDataset(torch.utils.data.Dataset):
    """In-memory tokenized dataset for GEKO."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _build_dataset(hf_dataset: Any, tokenizer: Any, max_len: int) -> _SimpleDataset:
    """
    Tokenize a HuggingFace dataset into a GEKO-compatible Dataset.
    Auto-detects common column formats (problem/solution, text, instruction/output, …).
    """
    cols = set(hf_dataset.column_names)

    def fmt(ex: dict) -> str:
        if 'problem' in cols and 'solution' in cols:
            return f"Problem: {ex['problem']}\n\nSolution: {ex['solution']}"
        if 'question' in cols and 'answer' in cols:
            return f"Question: {ex['question']}\n\nAnswer: {ex['answer']}"
        if 'instruction' in cols and 'output' in cols:
            return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        if 'instruction' in cols and 'response' in cols:
            return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
        if 'prompt' in cols and 'completion' in cols:
            return f"{ex['prompt']}{ex['completion']}"
        if 'text' in cols:
            return str(ex['text'])
        # Fallback: join all string values
        return " ".join(str(v) for v in ex.values() if isinstance(v, str))

    items = []
    for ex in hf_dataset:
        text = fmt(ex)
        enc = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        items.append({
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         enc["input_ids"].squeeze(0).clone(),
        })

    return _SimpleDataset(items)


# ══════════════════════════════════════════════════════════════════════════════
# Training thread
# ══════════════════════════════════════════════════════════════════════════════

def _train_thread(
    q: queue.Queue,
    stop_ev: threading.Event,
    model_name: str,
    dataset_name: str,
    split: str,
    n_samples: int,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    precision: str,
    max_len: int,
    freeze_conf: float,
    focus_conf: float,
    freeze_quality: float,
    use_curr: bool,
    prune_after: int,
    compile_model: bool,
    gradient_checkpointing: bool,
    use_8bit_optimizer: bool,
    enable_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_target_modules: str,
    lora_dropout: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    seed: int,
    output_dir_user: str,
    resume_path: str,
    eval_dataset_name: str,
    eval_split_eval: str,
):
    """Runs GEKO training in a background thread, pushing events to the queue."""
    try:
        # ── Resolve output directory ───────────────────────────────────────────
        output_dir = output_dir_user.strip() or f"./geko_run_{int(time.time())}"

        # ── Load model ────────────────────────────────────────────────────────
        q.put({'type': 'log', 'msg': f'Loading model: {model_name} …'})
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if hasattr(model, 'config'):
            model.config.pad_token_id = tokenizer.eos_token_id
        q.put({'type': 'log', 'msg': f'✓ {model_name} loaded  ({sum(p.numel() for p in model.parameters()):,} params)'})

        # ── Load dataset ──────────────────────────────────────────────────────
        q.put({'type': 'log', 'msg': f'Loading dataset: {dataset_name} [{split}] …'})
        from datasets import load_dataset
        raw = load_dataset(dataset_name, split=split)
        if n_samples == -1:
            # "All" — use the full dataset
            q.put({'type': 'log', 'msg': f'✓ {len(raw):,} samples loaded (full dataset)'})
        else:
            raw = raw.select(range(min(n_samples, len(raw))))
            q.put({'type': 'log', 'msg': f'✓ {len(raw):,} samples loaded'})
        actual_n = len(raw)   # resolved count used for step estimation below

        # ── Tokenize ──────────────────────────────────────────────────────────
        q.put({'type': 'log', 'msg': f'Tokenizing (max_length={max_len}) …'})
        dataset = _build_dataset(raw, tokenizer, max_len)
        q.put({'type': 'log', 'msg': f'✓ Tokenization done — {len(dataset):,} samples ready'})

        # ── Load eval dataset (optional) ──────────────────────────────────────
        eval_ds = None
        if eval_dataset_name.strip():
            q.put({'type': 'log', 'msg': f'Loading eval dataset: {eval_dataset_name} [{eval_split_eval}] …'})
            raw_eval = load_dataset(eval_dataset_name, split=eval_split_eval)
            raw_eval = raw_eval.select(range(min(500, len(raw_eval))))  # cap eval at 500 samples
            eval_ds = _build_dataset(raw_eval, tokenizer, max_len)
            q.put({'type': 'log', 'msg': f'✓ {len(eval_ds):,} eval samples ready'})

        # ── Build GEKO config ─────────────────────────────────────────────────
        use_bf16 = (precision == "BF16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = (precision == "FP16") and torch.cuda.is_available() and not use_bf16
        # On CPU, always fall back to FP32
        if not torch.cuda.is_available():
            use_bf16 = use_fp16 = False

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        q.put({'type': 'log', 'msg': f'Device: {device_str.upper()}  |  Precision: {precision}'})

        config = GEKOConfig(
            freeze_confidence  = freeze_conf,
            freeze_quality     = freeze_quality,
            focus_confidence   = focus_conf,
            use_curriculum     = use_curr,
            prune_frozen_after = prune_after,
            log_bucket_stats   = True,
        )

        # Estimate total optimizer steps so we can set sensible logging/save intervals
        n_batches_per_epoch = max(1, (actual_n + batch_size - 1) // batch_size)
        n_optimizer_steps   = max(1, (n_batches_per_epoch + grad_accum - 1) // grad_accum)
        # Log every optimizer step — guarantees chart data even on tiny datasets
        log_every = 1
        # Disable mid-training saves (save_at_end handles the final checkpoint);
        # setting save_steps > total steps avoids the spurious checkpoint-0 saves
        # that occur because global_step==0 at batch start satisfies 0 % save_steps == 0.
        save_every = n_optimizer_steps * epochs + 1

        # Emit total steps so the UI can compute ETA
        q.put({'type': 'total_steps', 'n': n_optimizer_steps * epochs})

        args = GEKOTrainingArgs(
            output_dir                  = output_dir,
            num_epochs                  = epochs,
            batch_size                  = batch_size,
            learning_rate               = lr,
            weight_decay                = weight_decay,
            warmup_steps                = warmup_steps,
            max_grad_norm               = max_grad_norm,
            seed                        = seed,
            gradient_accumulation_steps = grad_accum,
            bf16                        = use_bf16,
            fp16                        = use_fp16,
            gradient_checkpointing      = gradient_checkpointing,
            compile_model               = compile_model,
            use_8bit_optimizer          = use_8bit_optimizer,
            logging_steps               = log_every,
            save_steps                  = save_every,
            save_at_end                 = True,
            use_rich_ui                 = False,   # UI handled by _GradioUI
        )

        # ── LoRA (optional) ──────────────────────────────────────────────────
        lora_cfg = None
        if enable_lora:
            from geko.peft_utils import is_peft_available as _peft_ok
            if _peft_ok():
                from peft import LoraConfig, TaskType
                targets = [t.strip() for t in lora_target_modules.split(',') if t.strip()]
                if not targets:
                    targets = ["q_proj", "v_proj"]
                lora_cfg = LoraConfig(
                    task_type      = TaskType.CAUSAL_LM,
                    r              = lora_r,
                    lora_alpha     = lora_alpha,
                    lora_dropout   = lora_dropout,
                    target_modules = targets,
                )
                q.put({'type': 'log', 'msg': f'LoRA: r={lora_r}, α={lora_alpha}, targets={targets}'})
            else:
                q.put({'type': 'log', 'msg': '⚠ LoRA enabled but peft not installed — pip install peft'})

        trainer = GEKOTrainer(
            model         = model,
            train_dataset = dataset,
            tokenizer     = tokenizer,
            config        = config,
            args          = args,
            lora_config   = lora_cfg,
            eval_dataset  = eval_ds,
        )
        # ── Swap in the queue-based UI ────────────────────────────────────────
        trainer.ui = _GradioUI(q, stop_ev)

        # ── Resume from checkpoint (optional) ─────────────────────────────────
        if resume_path.strip():
            q.put({'type': 'log', 'msg': f'Resuming from: {resume_path.strip()} …'})
            trainer.load_checkpoint(resume_path.strip())

        q.put({'type': 'log', 'msg': '🚀 Training started!'})
        trainer.train()

    except StopIteration:
        q.put({'type': 'log', 'msg': '⏹ Training stopped by user.'})
        q.put({'type': 'stopped'})

    except Exception:
        err = traceback.format_exc()
        q.put({'type': 'error', 'msg': err})

    finally:
        q.put({'type': 'finished'})


# ══════════════════════════════════════════════════════════════════════════════
# Chart builder  (creates Plotly figures from accumulated data)
# ══════════════════════════════════════════════════════════════════════════════

_CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(8,8,18,1)',
    font=dict(family="'DM Mono', 'JetBrains Mono', monospace", size=11, color='#5A6180'),
    margin=dict(l=52, r=20, t=44, b=44),
    height=380,
    xaxis=dict(gridcolor='#1A1E2E', gridwidth=1, zeroline=False, showline=False,
               tickfont=dict(size=10, color='#3A4160')),
    yaxis=dict(gridcolor='#1A1E2E', gridwidth=1, zeroline=False, showline=False,
               tickfont=dict(size=10, color='#3A4160')),
)


def _make_loss_fig(steps: List[int], losses: List[float]):
    """Return a Plotly figure for the loss curve."""
    fig = go.Figure()
    if steps:
        fig.add_trace(go.Scatter(
            x=steps, y=losses,
            mode='lines+markers',
            line=dict(color='#00FF88', width=2),
            marker=dict(size=5, color='#00FF88',
                        line=dict(width=1, color='#09090F')),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.06)',
            name='Loss',
        ))
    layout = dict(
        **_CHART_LAYOUT,
        title=dict(text="Training Loss", font=dict(size=12, color='#3A4160')),
        xaxis_title="Step",
        yaxis_title="Loss",
    )
    fig.update_layout(**layout)
    if not steps:
        fig.add_annotation(
            text="Loss curve will appear once training starts",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color='#2A2E42'),
        )
    return fig


def _make_bucket_fig(bucket_history: list):
    """Return a Plotly bar chart showing the latest epoch's bucket distribution."""
    fig = go.Figure()
    layout = dict(
        **_CHART_LAYOUT,
        showlegend=False,
    )

    if not bucket_history:
        fig.update_layout(**layout,
            title=dict(text="Bucket Distribution", font=dict(size=12, color='#3A4160')))
        fig.add_annotation(
            text="Bucket stats will appear after epoch 1",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color='#2A2E42'),
        )
        return fig

    last = bucket_history[-1]
    names  = ["FREEZE", "LIGHT", "FOCUS", "HARD"]
    values = [last['freeze'], last['light'], last['focus'], last['hard']]
    colors = ['#4C8BF5', '#22D3EE', '#00FF88', '#FF4455']
    compute_saved = last['freeze'] / max(last['total'], 1)

    fig.add_trace(go.Bar(
        x=names, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:,}" for v in values],
        textposition='outside',
        textfont=dict(size=11, color='#8B95B0'),
    ))
    fig.update_layout(**layout,
        title=dict(
            text=f"Epoch {last['epoch']} Buckets  ·  ~{compute_saved:.0%} compute saved",
            font=dict(size=12, color='#3A4160'),
        ),
        yaxis_title="Samples",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Misc helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_eta(secs: float) -> str:
    secs = int(secs)
    if secs < 60:   return f"{secs}s"
    if secs < 3600: return f"{secs // 60}m {secs % 60:02d}s"
    return f"{secs // 3600}h {(secs % 3600) // 60:02d}m"


def _make_summary_html(loss, compute_saved, out_dir, total_steps, step) -> str:
    if loss is None:
        return ""
    cs = f"{compute_saved:.0%}" if compute_saved is not None else "—"
    return (
        '<div style="margin-top:8px; padding:14px 18px; border-radius:8px;'
        ' border:1px solid #1a2a1a; background:rgba(0,255,136,0.04);">'
        '<div style="font-size:12px; font-weight:600; color:#00FF88; margin-bottom:8px;">'
        'Training Complete ✓'
        '</div>'
        '<div style="display:flex; gap:32px; font-size:12px; opacity:0.7;">'
        f'<span>Loss <b>{loss:.4f}</b></span>'
        f'<span>Compute saved <b>{cs}</b></span>'
        f'<span>Steps <b>{step:,}</b></span>'
        f'<span>Saved to <b>{out_dir}</b></span>'
        '</div>'
        '</div>'
    )


def _hw_info() -> str:
    parts = [f"PyTorch {torch.__version__}"]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        parts.append(f"{name}  ·  {vram} GB VRAM")
        if torch.cuda.is_bf16_supported():
            parts.append("BF16 ✓")
    else:
        parts.append("CPU only — BF16/FP16 disabled")
    return (
        '<div style="padding:2px 8px 8px; font-size:11px; opacity:0.4;'
        ' display:flex; gap:16px; flex-wrap:wrap;">'
        + "".join(f"<span>{p}</span>" for p in parts)
        + "</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gradio generator  (streams updates from queue → browser)
# ══════════════════════════════════════════════════════════════════════════════

def _run_training(
    model_name, dataset_name, eval_dataset_name, eval_split_eval,
    split, n_samples, precision, max_len,
    epochs, lr, batch_size, grad_accum,
    freeze_conf, focus_conf, freeze_quality, use_curr, prune_after,
    compile_model, gradient_checkpointing, use_8bit_optimizer,
    enable_lora, lora_r, lora_alpha, lora_target_modules, lora_dropout,
    warmup_steps, weight_decay, max_grad_norm, seed,
    outdir_user, resume_path,
):
    """
    Gradio generator function — yields component updates as training progresses.
    Yields a 12-tuple: (status, loss_fig, bucket_fig,
                        epoch_str, step_str, compute_str, loss_str,
                        speed_str, eta_str, eval_loss_str,
                        summary_html, save_str)
    """
    global _stop_event, _current_thread

    # Validate deps
    if not PLOTLY_AVAILABLE:
        yield ("Error: plotly not installed. Run: pip install plotly",
               None, None, "—", "—", "—", "—", "—", "—", "—", "", "—")
        return

    _stop_event = threading.Event()
    q: queue.Queue = queue.Queue()

    _current_thread = threading.Thread(
        target=_train_thread,
        args=(
            q, _stop_event,
            model_name, dataset_name, split,
            (-1 if str(n_samples).strip().lower() == "all" else int(n_samples)),
            int(epochs), float(lr), int(batch_size), int(grad_accum),
            precision, int(max_len),
            float(freeze_conf), float(focus_conf), float(freeze_quality),
            bool(use_curr), int(prune_after),
            bool(compile_model), bool(gradient_checkpointing), bool(use_8bit_optimizer),
            bool(enable_lora), int(lora_r), int(lora_alpha),
            str(lora_target_modules), float(lora_dropout),
            int(warmup_steps), float(weight_decay), float(max_grad_norm), int(seed),
            str(outdir_user), str(resume_path),
            str(eval_dataset_name), str(eval_split_eval),
        ),
        daemon=True,
    )
    _current_thread.start()

    # Accumulated state
    loss_steps:     List[int]   = []
    loss_vals:      List[float] = []
    bucket_history: list        = []
    current_epoch  = 0
    total_epochs   = int(epochs)
    compute_saved: Optional[float] = None
    current_step   = 0                      # counts every 'step' event (one per batch)
    latest_batch_loss: Optional[float] = None
    avg_epoch_loss:    Optional[float] = None  # epoch-average loss from epoch_loss events
    training_started = False                # set True on first 'step' or 'partition' event
    loading_status   = "Loading…"           # surfaces trainer log msgs before first batch
    error_status     = "Error"              # stores first meaningful line of traceback
    finished   = False
    error_flag = False

    # Phase 2 state
    total_steps_expected: Optional[int] = None
    _step_times: deque = deque(maxlen=20)   # wall-clock timestamps of last 20 steps
    eval_loss_val: Optional[float] = None

    def _status_cls(status: str) -> str:
        # Use closure over error_flag / training_started so arbitrary strings work
        if error_flag or status.startswith("Error"):  return "status-error"
        if status.startswith("Done"):                 return "status-done"
        if status.startswith("Stopped"):              return "status-done"
        if not training_started:                      return "status-training"  # loading msgs
        if status == "Training…":                     return "status-training"
        return "status-ready"

    def _emit(status: str):
        epoch_str   = f"{current_epoch + 1} / {total_epochs}" if training_started else "—"
        step_str    = str(current_step) if current_step > 0 else "—"
        compute_str = f"{compute_saved:.0%}" if compute_saved is not None else "—"
        # Prefer epoch-average loss (more stable) over latest batch loss
        _loss = (avg_epoch_loss if avg_epoch_loss is not None else
                 latest_batch_loss if latest_batch_loss is not None else
                 (loss_vals[-1] if loss_vals else None))
        loss_str = f"{_loss:.4f}" if _loss is not None else "—"

        # Steps/sec + ETA from rolling 20-step window
        if len(_step_times) >= 2:
            elapsed = _step_times[-1] - _step_times[0]
            sps = (len(_step_times) - 1) / elapsed if elapsed > 0 else 0
            speed_str = f"{sps:.1f} step/s"
            if total_steps_expected and sps > 0:
                remaining = max(0, total_steps_expected - current_step)
                eta_str = _fmt_eta(remaining / sps)
            else:
                eta_str = "—"
        else:
            speed_str = "—"
            eta_str   = "—"

        eval_loss_str = f"{eval_loss_val:.4f}" if eval_loss_val is not None else "—"

        # Save path display — use user-provided dir if it exists
        _od = outdir_user.strip()
        save_str = _od if (_od and os.path.exists(_od)) else "—"

        # Training summary card (only after successful completion)
        if finished and not error_flag and not (_stop_event and _stop_event.is_set()):
            summary = _make_summary_html(avg_epoch_loss, compute_saved, _od or "(auto)",
                                         total_steps_expected, current_step)
        else:
            summary = ""

        try:    lfig = _make_loss_fig(loss_steps, loss_vals)
        except: lfig = go.Figure()
        try:    bfig = _make_bucket_fig(bucket_history)
        except: bfig = go.Figure()
        status_update = gr.update(value=status, elem_classes=[_status_cls(status)])
        return (status_update, lfig, bfig,
                epoch_str, step_str, compute_str, loss_str,
                speed_str, eta_str, eval_loss_str,
                summary, save_str)

    # Yield an initial "Loading…" state immediately
    yield _emit("Loading…")

    while not finished:
        # Drain queue in bursts for smooth UI updates
        burst_start = time.time()

        while time.time() - burst_start < 0.4:
            try:
                ev = q.get(timeout=0.05)
            except queue.Empty:
                if not _current_thread.is_alive():
                    finished = True
                break

            t = ev.get('type', '')

            if t == 'step':
                # Fired every batch — primary step/loss counter and chart data source
                current_step += 1
                latest_batch_loss = ev.get('loss')
                training_started = True
                _step_times.append(time.time())
                if latest_batch_loss is not None:
                    loss_steps.append(current_step)
                    loss_vals.append(latest_batch_loss)

            elif t == 'step_loss':
                pass  # deduped per optimizer step; no log textbox to display in

            elif t == 'partition':
                bucket_history.append(ev)
                current_epoch = ev['epoch']
                compute_saved = ev['freeze'] / max(ev['total'], 1)
                training_started = True

            elif t == 'epoch_loss':
                # Track epoch-average loss for display — do NOT update current_epoch here.
                # Trainer fires epoch_loss(epoch+1) after the epoch ends (1-indexed), so
                # using it for current_epoch would show "4 / 3" after the final epoch.
                # Epoch tracking is owned entirely by partition events (0-indexed, fires at
                # the START of each epoch before any batches run).
                avg_epoch_loss = ev.get('loss')

            elif t == 'total_steps':
                total_steps_expected = ev.get('n')

            elif t == 'eval_loss':
                eval_loss_val = ev.get('loss')

            elif t == 'log':
                # Surface trainer messages in the status bar during the loading phase
                # (model download, dataset load, tokenisation). Once the first batch runs,
                # status switches to "Training…" and these messages are silently dropped
                # (the log textbox was removed in favour of dominant charts).
                if not training_started:
                    loading_status = ev['msg']

            elif t == 'error':
                # Extract the first meaningful line of the traceback for the status bar
                raw_err = ev.get('msg', 'Unknown error')
                first_line = next(
                    (l.strip() for l in raw_err.splitlines()
                     if l.strip() and not l.strip().startswith(('File ', 'Traceback'))),
                    'Training failed'
                )
                error_status = f"Error: {first_line[:120]}"
                error_flag = True
                finished = True

            elif t == 'done':
                # Extract final efficiency data from the summary event so the stat boxes
                # show the definitive values rather than the last mid-epoch snapshot.
                report = ev.get('report') or {}
                result = ev.get('result') or {}
                cs_str = report.get('compute_saved_percent', '')
                if cs_str:
                    try:
                        compute_saved = float(cs_str.strip('%')) / 100
                    except ValueError:
                        pass
                final_loss = result.get('total_loss')
                if final_loss is not None:
                    avg_epoch_loss = final_loss
                finished = True

            elif t in ('stopped', 'finished'):
                finished = True

        # Yield update to Gradio
        if error_flag:
            status = error_status
        elif finished:
            status = "Stopped" if (_stop_event and _stop_event.is_set()) else "Done ✓"
        elif training_started:
            status = "Training…"
        else:
            status = loading_status
        yield _emit(status)

    # Final yield with definitive status
    if error_flag:
        final = error_status
    elif _stop_event and _stop_event.is_set():
        final = "Stopped"
    else:
        final = "Done ✓"
    yield _emit(final)


def _kill_training_thread(thread: Optional[threading.Thread]) -> None:
    """
    Asynchronously raise SystemExit inside a running thread using the CPython
    C-API. This causes the thread to terminate at the very next Python bytecode
    instruction — no waiting for the current batch to finish.
    """
    if thread is None or not thread.is_alive():
        return
    tid = thread.ident
    if tid is None:
        return
    # PyThreadState_SetAsyncExc schedules an exception to be raised in the
    # target thread the next time the interpreter checks for async events.
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(SystemExit),
    )


def _stop_training():
    """Called by the Stop button — immediately kills the training thread."""
    global _stop_event, _current_thread
    # Set the cooperative stop flag first (lets the finally block run cleanly)
    if _stop_event is not None:
        _stop_event.set()
    # Force-raise SystemExit inside the training thread for instant termination
    _kill_training_thread(_current_thread)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio app layout
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Hide Gradio chrome ─────────────────────────────────────── */
footer, .built-with { display: none !important; }
.gradio-container { max-width: 100% !important; }

/* ── Status color coding ────────────────────────────────────── */
.status-ready    textarea { color: var(--body-text-color-subdued) !important; }
.status-training textarea { color: #00FF88 !important; animation: geko-blink 2s ease-in-out infinite; }
.status-done     textarea { color: #34D399 !important; }
.status-error    textarea { color: #ff4d4d !important; }

@keyframes geko-blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.6; }
}

/* ── Stat boxes ─────────────────────────────────────────────── */
.stat-box textarea {
  text-align: center !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  color: #00FF88 !important;
}

/* ── Training log ───────────────────────────────────────────── */
.training-log textarea {
  font-family: 'SF Mono','Monaco','Menlo','Consolas',monospace !important;
  font-size: 12px !important;
  line-height: 1.7 !important;
}

/* ── Stop button ────────────────────────────────────────────── */
.stop-btn button { color: #ff4d4d !important; }
.stop-btn button:disabled { opacity: 0.35 !important; }

/* ── Gecko animation ────────────────────────────────────────── */
@keyframes geko-pulse {
  0%,100% { transform: scale(1) rotate(0deg); }
  50%      { transform: scale(1.10) rotate(-5deg); }
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { border-radius: 3px; }
"""

def create_app() -> "gr.Blocks":
    if not GRADIO_AVAILABLE:
        raise ImportError(
            "gradio is required to run the GEKO app. Install it with:\n"
            "    pip install gekolib[app]\n"
            "or:\n"
            "    pip install gradio plotly datasets transformers"
        )

    from geko.core import GEKO_AGGRESSIVE, GEKO_BALANCED, GEKO_CONSERVATIVE

    with gr.Blocks(title="GEKO — Fine-Tuning App", elem_id="geko-app") as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="padding: 18px 8px 4px; display: flex; align-items: center; gap: 14px;">
          <span style="font-size:34px; line-height:1; display:inline-block;
            animation: geko-pulse 3s ease-in-out infinite;">🦎</span>
          <div>
            <span style="font-size:26px; font-weight:800; color:#00FF88; letter-spacing:-0.01em;">GEKO</span>
            <span style="margin-left:10px; font-size:11px; opacity:0.4; letter-spacing:0.06em; text-transform:uppercase; font-weight:500;">
              v0.3.1 &nbsp;·&nbsp; Gradient-Efficient Knowledge Optimization
            </span>
          </div>
          <div style="margin-left:auto; font-size:12px; opacity:0.45;">
            <a href="https://github.com/Abd0r/GEKO" target="_blank"
               style="text-decoration:none; margin-right:14px;">GitHub</a>
            <a href="https://pypi.org/project/gekolib/" target="_blank"
               style="text-decoration:none;">PyPI</a>
          </div>
        </div>
        """)

        # ── Hardware info (auto-populated at page load) ─────────────────────
        hw_html = gr.HTML("")

        # ── Row 1: Model + Dataset + Eval Dataset + Eval Split ──────────────
        with gr.Row():
            model_in = gr.Textbox(
                label="Model (HuggingFace ID or local path)",
                value="gpt2",
                placeholder="gpt2  ·  meta-llama/Llama-3.2-1B  ·  …",
                scale=1,
            )
            dataset_in = gr.Textbox(
                label="Dataset (HuggingFace ID)",
                value="open-r1/OpenR1-Math-220k",
                placeholder="open-r1/OpenR1-Math-220k  ·  tatsu-lab/alpaca  ·  …",
                scale=1,
            )
            eval_dataset_in = gr.Textbox(
                label="Eval Dataset  (optional HF ID)",
                value="",
                placeholder="leave blank to skip eval  ·  tatsu-lab/alpaca  ·  …",
                scale=1,
            )
            eval_split_in = gr.Dropdown(
                choices=["validation", "test", "train"],
                value="validation", label="Eval Split",
                allow_custom_value=True, scale=1,
            )

        # ── Row 2: Data options + precision ────────────────────────────────
        with gr.Row():
            split_in = gr.Dropdown(
                choices=["train", "validation", "test"],
                value="train", label="Split",
                allow_custom_value=True,
            )
            n_samples_in = gr.Dropdown(
                choices=["100", "500", "1000", "5000", "10000", "50000", "100000", "All"],
                value="500", label="Max Samples",
                allow_custom_value=True,
            )
            prec_in = gr.Dropdown(
                choices=["BF16", "FP16", "FP32"],
                value="BF16", label="Precision",
            )
            maxlen_in = gr.Dropdown(
                choices=["128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768"],
                value="512", label="Max Seq Length",
                allow_custom_value=True,
            )

        # ── Row 3: Training hyperparams ─────────────────────────────────────
        with gr.Row():
            epochs_in = gr.Dropdown(
                choices=["1", "2", "3", "5", "10", "20"],
                value="3", label="Epochs",
                allow_custom_value=True,
            )
            lr_in = gr.Dropdown(
                choices=["1e-3", "5e-4", "1e-4", "5e-5", "3e-5", "1e-5", "5e-6"],
                value="3e-5", label="Learning Rate",
                allow_custom_value=True,
            )
            batch_in = gr.Dropdown(
                choices=["1", "4", "8", "16", "32", "64", "128"],
                value="8", label="Batch Size",
                allow_custom_value=True,
            )
            accum_in = gr.Dropdown(
                choices=["1", "2", "4", "8", "16", "32"],
                value="4", label="Grad Accum Steps",
                allow_custom_value=True,
            )

        # ── Efficiency accordion ────────────────────────────────────────────
        with gr.Accordion("⚡ Efficiency", open=False):
            with gr.Row():
                compile_in = gr.Checkbox(
                    value=False,
                    label="torch.compile  (20–50% throughput boost, PyTorch 2.0+)",
                    scale=3,
                )
                grad_ckpt_in = gr.Checkbox(
                    value=False,
                    label="Gradient Checkpointing  (~4× activation memory saving)",
                    scale=3,
                )
                bnb_in = gr.Checkbox(
                    value=False,
                    label="8-bit Optimizer  (~2× optimizer memory, needs bitsandbytes)",
                    scale=3,
                )

        # ── LoRA / PEFT accordion ───────────────────────────────────────────
        with gr.Accordion("🔗 LoRA / PEFT", open=False):
            with gr.Row():
                lora_en_in = gr.Checkbox(value=False, label="Enable LoRA", scale=1)
                lora_r_in = gr.Dropdown(
                    choices=["4", "8", "16", "32", "64", "128"],
                    value="16", label="Rank (r)", allow_custom_value=True, scale=1,
                )
                lora_alpha_in = gr.Dropdown(
                    choices=["8", "16", "32", "64", "128"],
                    value="32", label="Alpha", allow_custom_value=True, scale=1,
                )
                lora_dropout_in = gr.Slider(
                    minimum=0.0, maximum=0.3, value=0.05, step=0.01,
                    label="Dropout", scale=1,
                )
            lora_targets_in = gr.Textbox(
                label="Target Modules  (comma-separated — blank = auto: q_proj,v_proj)",
                value="q_proj,v_proj",
                placeholder="q_proj,v_proj  ·  c_attn  ·  query_key_value  ·  …",
            )

        # ── GEKO Config accordion ───────────────────────────────────────────
        with gr.Accordion("GEKO Config", open=False):
            with gr.Row():
                preset_in = gr.Dropdown(
                    choices=["Custom", "Balanced", "Aggressive", "Conservative"],
                    value="Custom", label="Preset", scale=1,
                )
            with gr.Row():
                freeze_in = gr.Slider(
                    minimum=0.50, maximum=0.99, value=0.85, step=0.01,
                    label="Freeze Threshold  (above → skip sample)",
                    scale=3,
                )
                focus_in = gr.Slider(
                    minimum=0.20, maximum=0.79, value=0.50, step=0.01,
                    label="Focus Threshold  (below → FOCUS  ·  above → HARD)",
                    scale=3,
                )
                freeze_quality_in = gr.Slider(
                    minimum=0.50, maximum=0.99, value=0.80, step=0.01,
                    label="Freeze Quality  (Q-value gate for FREEZE)",
                    scale=2,
                )
            with gr.Row():
                curr_in = gr.Checkbox(value=True, label="Mountain Curriculum", scale=1)
                prune_in = gr.Slider(
                    minimum=0, maximum=10, value=0, step=1,
                    label="Prune Frozen After N Epochs  (0 = off)",
                    scale=2,
                )

        # ── Advanced accordion ──────────────────────────────────────────────
        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                warmup_in = gr.Dropdown(
                    choices=["0", "50", "100", "200", "500", "1000"],
                    value="100", label="Warmup Steps", allow_custom_value=True,
                )
                wdecay_in = gr.Dropdown(
                    choices=["0.0", "0.001", "0.01", "0.05", "0.1"],
                    value="0.01", label="Weight Decay", allow_custom_value=True,
                )
                maxnorm_in = gr.Dropdown(
                    choices=["0.5", "1.0", "2.0", "5.0"],
                    value="1.0", label="Max Grad Norm", allow_custom_value=True,
                )
                seed_in = gr.Dropdown(
                    choices=["0", "42", "123", "1337", "2024", "2025"],
                    value="42", label="Seed", allow_custom_value=True,
                )

        # ── Output directory + resume checkpoint ────────────────────────────
        outdir_in = gr.Textbox(
            label="Output Directory",
            value="./geko_output",
            placeholder="./geko_output  ·  /absolute/path/to/save  ·  …",
        )
        resume_in = gr.Textbox(
            label="Resume from Checkpoint  (optional — leave blank to start fresh)",
            value="",
            placeholder="./geko_output/checkpoint-1000",
        )

        # ── Bucket legend ───────────────────────────────────────────────────
        gr.HTML("""
        <div style="padding: 6px 2px 2px; font-size: 11px; opacity: 0.55; display:flex; gap:20px; flex-wrap:wrap;">
          <span style="color:#4C8BF5;">● FREEZE &mdash; skip mastered</span>
          <span style="color:#22D3EE;">● LIGHT &mdash; low priority</span>
          <span style="color:#00FF88;">● FOCUS &mdash; medium priority</span>
          <span style="color:#ff4d4d;">● HARD &mdash; 3&times; weight</span>
        </div>
        """)

        # ── Action buttons ──────────────────────────────────────────────────
        with gr.Row():
            start_btn = gr.Button("▶  Start Training", variant="primary", scale=4)
            stop_btn  = gr.Button("■  Stop", variant="secondary", scale=1,
                                  interactive=False, elem_classes=["stop-btn"])
            clear_btn = gr.Button("✕  Clear", variant="secondary", scale=1)

        # ── Output: status + stats ──────────────────────────────────────────
        status_out = gr.Textbox(
            label="Status",
            value="Ready — configure above and click Start Training",
            lines=1, max_lines=1,
            interactive=False,
            elem_classes=["status-ready"],
        )
        with gr.Row():
            epoch_out = gr.Textbox(
                label="Epoch", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            step_out = gr.Textbox(
                label="Step", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            compute_out = gr.Textbox(
                label="Compute Saved", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            loss_out = gr.Textbox(
                label="Latest Loss", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            speed_out = gr.Textbox(
                label="Steps/sec", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            eta_out = gr.Textbox(
                label="ETA", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )
            eval_loss_out = gr.Textbox(
                label="Eval Loss", value="—", lines=1, max_lines=1,
                interactive=False, elem_classes=["stat-box"],
            )

        # ── Charts: full width, side by side ───────────────────────────────
        with gr.Row():
            loss_plot   = gr.Plot(label="Loss Curve",
                                  value=_make_loss_fig([], []))
            bucket_plot = gr.Plot(label="Bucket Distribution",
                                  value=_make_bucket_fig([]))

        save_out = gr.Textbox(
            label="Output Directory", value="—",
            lines=1, max_lines=1, interactive=False,
        )

        summary_html = gr.HTML("")

        # ── Wiring ──────────────────────────────────────────────────────────
        all_inputs = [
            model_in, dataset_in, eval_dataset_in, eval_split_in,
            split_in, n_samples_in, prec_in, maxlen_in,
            epochs_in, lr_in, batch_in, accum_in,
            freeze_in, focus_in, freeze_quality_in, curr_in, prune_in,
            compile_in, grad_ckpt_in, bnb_in,
            lora_en_in, lora_r_in, lora_alpha_in, lora_targets_in, lora_dropout_in,
            warmup_in, wdecay_in, maxnorm_in, seed_in,
            outdir_in, resume_in,
        ]
        all_outputs = [
            status_out, loss_plot, bucket_plot,
            epoch_out, step_out, compute_out, loss_out,
            speed_out, eta_out, eval_loss_out,
            summary_html, save_out,
        ]

        # Preset dropdown → auto-fill freeze / focus / freeze_quality sliders
        def _apply_preset(preset):
            _PRESETS = {
                "Aggressive":   GEKO_AGGRESSIVE,
                "Balanced":     GEKO_BALANCED,
                "Conservative": GEKO_CONSERVATIVE,
            }
            cfg = _PRESETS.get(preset)
            if cfg is None:
                return gr.update(), gr.update(), gr.update()
            return (gr.update(value=cfg.freeze_confidence),
                    gr.update(value=cfg.focus_confidence),
                    gr.update(value=cfg.freeze_quality))

        preset_in.change(
            fn=_apply_preset, inputs=[preset_in],
            outputs=[freeze_in, focus_in, freeze_quality_in],
        )

        # Clear / Reset all outputs
        def _on_clear():
            empty_loss   = _make_loss_fig([], [])
            empty_bucket = _make_bucket_fig([])
            ready = gr.update(
                value="Ready — configure above and click Start Training",
                elem_classes=["status-ready"],
            )
            dash = "—"
            return (ready, empty_loss, empty_bucket,
                    dash, dash, dash, dash,
                    dash, dash, dash,
                    "", dash)

        clear_btn.click(fn=_on_clear, inputs=[], outputs=all_outputs)

        def _on_start():
            return gr.update(interactive=False), gr.update(interactive=True)

        def _on_finish():
            return gr.update(interactive=True), gr.update(interactive=False)

        start_event = start_btn.click(fn=_run_training, inputs=all_inputs, outputs=all_outputs)
        start_btn.click(fn=_on_start, inputs=[], outputs=[start_btn, stop_btn])
        start_event.then(fn=_on_finish, inputs=[], outputs=[start_btn, stop_btn])
        stop_btn.click(fn=_stop_training, inputs=[], outputs=[])

        # Hardware info auto-populated at page load
        demo.load(fn=_hw_info, outputs=[hw_html])

    return demo


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Launch the GEKO web app."""
    print("=" * 55)
    print("  GEKO Fine-Tuning App")
    print("=" * 55)

    if not GRADIO_AVAILABLE:
        print("\n[ERROR] gradio is not installed.")
        print("Install it with:\n    pip install gekolib[app]")
        raise SystemExit(1)

    if not PLOTLY_AVAILABLE:
        print("\n[ERROR] plotly is not installed.")
        print("Install it with:\n    pip install plotly")
        raise SystemExit(1)

    demo = create_app()
    print("\n  Opening in your browser at http://localhost:7860")
    print("  Press Ctrl+C to stop the server.\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.slate,
        ),
    )


if __name__ == "__main__":
    main()
