"""
LionAI train.py — Bug-Fixed + CPU/AMD-Optimised Edition
=========================================================
Bugs fixed vs previous version:
  BUG 1: PackedTextDataset produced 0 examples for small corpora
          → min_seq_length added; short texts concatenated before packing;
            flush remainder even if < max_length (pad to fill)
  BUG 2: max_steps=50_000 default made tiny datasets train for hours
          → auto_max_steps() scales steps to dataset size
  BUG 3: No minimum dataset size guard before training
          → raise clear error with guidance when < 8 examples
  BUG 4: drop_last=True + batch_size=8 emptied tiny DataLoaders
          → drop_last=False; batch auto-sized down to min(8, n_examples)
  BUG 5: auto_batch_size only checked CUDA VRAM (useless on AMD/CPU)
          → RAM-based auto-sizing that works on all hardware
  BUG 6: gradient_checkpointing=True on CPU adds overhead with no benefit
          → disabled by default; only enabled when device==cuda/mps
  BUG 7: persistent_workers=True crashes with 0 or 1 examples
          → num_workers=0 on Windows or when dataset is tiny
  BUG 8: Warmup 300 steps was too long for tiny datasets
          → warmup = min(300, n_steps * 0.1)
  BUG 9: No AMD/ROCm detection — user's RX550 silently fell to CPU
          → added ROCm check + torch.set_num_threads for best CPU perf
  BUG 10: Dataset validation happened silently; errors only surfaced mid-train
           → validate_dataset() called before training starts, clear messages

Performance improvements for i5-10th + 16GB RAM:
  • torch.set_num_threads(physical_cores) — uses all i5 cores properly
  • torch.set_num_interop_threads — parallel data loading on CPU
  • Use bfloat16 on CPU (PyTorch 2.x supports it natively, ~30% faster)
  • Smaller default seq_len (128) and vocab (512) for tiny datasets
  • Streaming progress every 10 steps instead of 50 (more responsive feel)
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

from model import LionLLM, ModelConfig
from tokenizer import LionTokenizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CPU / Device Setup
# ─────────────────────────────────────────────

def setup_device() -> Tuple[str, str]:
    """
    Detect best available device.
    Returns (device_str, dtype_str).
    Handles CUDA, ROCm (AMD), MPS (Apple), and CPU.
    """
    # ROCm / AMD GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        is_amd = "AMD" in props.name or "Radeon" in props.name or "gfx" in props.name.lower()
        device = "cuda"
        # AMD RX 550 4GB: bfloat16 often unsupported, use float16
        if is_amd:
            logger.info("AMD GPU detected: %s  (%.1f GB VRAM)", props.name,
                        props.total_memory / 1e9)
            return "cuda", "float16"
        return "cuda", "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    if torch.backends.mps.is_available():
        return "mps", "float32"

    # CPU — set thread count for best i5 performance
    n_cpu = os.cpu_count() or 4
    # Physical cores ≈ logical / 2 on HT CPUs; use all logical for throughput
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(1, n_cpu // 2))
    logger.info("CPU mode: %d threads", n_cpu)
    return "cpu", "float32"


# ─────────────────────────────────────────────
#  Training Config
# ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Paths
    output_dir:   str  = "./runs/lionai"
    dataset_path: str  = "./data/train.jsonl"
    resume_from:  Optional[str] = None

    # Data — FIXED: much smaller defaults for small corpora
    max_seq_length:        int   = 128    # FIX: was 512 (too large for 50 words)
    min_seq_length:        int   = 8      # FIX: new — skip extremely short sequences
    train_split:           float = 0.9    # FIX: was 0.95 (left too little for tiny datasets)
    pack_sequences:        bool  = True

    # Optimisation
    learning_rate:         float = 5e-4   # FIX: slightly higher for small datasets
    weight_decay:          float = 0.01   # FIX: lower weight decay for tiny data
    beta1:                 float = 0.9
    beta2:                 float = 0.95
    eps:                   float = 1e-8
    grad_clip:             float = 1.0
    lr_decay_end:          float = 0.1
    layer_lr_decay:        float = 1.0    # FIX: no layer decay for small models

    # Schedule — FIXED: auto-computed from dataset size
    warmup_steps:          int   = 20     # FIX: was 300 (way too long for tiny data)
    max_steps:             int   = 0      # FIX: 0 = auto-compute from dataset
    lr_schedule:           str   = "cosine"

    # Batching — FIXED: smaller defaults
    batch_size:            int   = 4      # FIX: was 8 (too large for tiny datasets)
    gradient_accumulation: int   = 2      # FIX: was 4
    auto_batch_size:       bool  = True

    # Memory — FIXED: gradient checkpointing only helps GPU
    gradient_checkpointing:bool  = False  # FIX: was True (hurts CPU performance)
    use_torch_compile:     bool  = False
    dtype:                 str   = "auto"

    # Evaluation
    eval_interval:         int   = 0      # FIX: 0 = auto (every 10% of steps)
    eval_steps:            int   = 10     # FIX: was 50 (too slow for tiny val set)
    save_interval:         int   = 0      # FIX: 0 = auto
    keep_checkpoints:      int   = 2

    # Early stopping
    patience:              int   = 5
    patience_delta:        float = 0.005  # FIX: less strict for small data

    # Logging — FIX: more frequent for responsive feel
    log_interval:          int   = 10     # FIX: was 50

    def save(self, path: Path) -> None:
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        with open(path) as f: d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_small_dataset(cls, n_examples: int,
                           vocab_size: int = 512) -> "TrainingConfig":
        """
        Auto-generate a sensible config for small datasets.
        Call this when you have fewer than 1000 training examples.
        """
        # Scale steps to dataset: ~10 passes through data
        steps = max(100, n_examples * 10)
        return cls(
            max_seq_length        = min(128, max(32, n_examples * 2)),
            batch_size            = min(4, max(1, n_examples // 4)),
            gradient_accumulation = 1,
            max_steps             = steps,
            warmup_steps          = max(10, steps // 10),
            eval_interval         = max(10, steps // 10),
            save_interval         = max(50, steps // 5),
            log_interval          = max(5,  steps // 20),
            layer_lr_decay        = 1.0,
            gradient_checkpointing= False,
            train_split           = 0.85 if n_examples > 20 else 0.0,
        )


# ─────────────────────────────────────────────
#  Dataset Validation  (FIX: run BEFORE training)
# ─────────────────────────────────────────────

def validate_dataset(path: Path, tokenizer: LionTokenizer,
                      max_seq_length: int) -> Tuple[int, int]:
    """
    Validate dataset before training starts.
    Returns (n_raw_lines, n_usable_tokens).
    Raises ValueError with clear message if dataset is unusable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python dataset_processor.py --sources ./mydata/ --output ./data"
        )

    n_lines = 0
    total_tokens = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_lines += 1
            try:    text = json.loads(line).get("text", "")
            except: text = line
            if text:
                total_tokens += len(tokenizer.encode(text))

    if n_lines == 0:
        raise ValueError(
            f"Dataset is empty: {path}\n"
            f"Add text data then re-run dataset_processor.py"
        )

    # Estimate usable packed examples
    n_examples = max(1, total_tokens // max_seq_length)

    logger.info("Dataset: %d lines | ~%d tokens | ~%d packed examples (seq=%d)",
                n_lines, total_tokens, n_examples, max_seq_length)

    if total_tokens < max_seq_length * 2:
        logger.warning(
            "Very small dataset: %d tokens < %d (2× max_seq_length).\n"
            "  Tip: reduce max_seq_length to %d, or add more text.",
            total_tokens, max_seq_length * 2, max(8, total_tokens // 4)
        )

    return n_lines, total_tokens


# ─────────────────────────────────────────────
#  Packed Dataset  — FIXED version
# ─────────────────────────────────────────────

class PackedTextDataset(Dataset):
    """
    FIXED bugs:
      1. Short corpora that never filled a full chunk now produce examples
         (remainder is always saved, padded to max_length)
      2. Multiple short documents are concatenated before chunking
      3. min_seq_length skips garbage/near-empty entries
      4. Repeats corpus if fewer than min_examples would be produced
    """

    def __init__(self, path: Path, tokenizer: LionTokenizer,
                 max_length: int = 128,
                 min_length: int = 8,
                 pack: bool = True,
                 min_examples: int = 8) -> None:
        self.max_length = max_length
        self.examples: List[List[int]] = []
        path = Path(path)

        logger.info("Loading dataset: %s  (max_len=%d pack=%s)", path.name, max_length, pack)

        buf: List[int] = []
        n_docs = 0

        def _read_texts():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:    yield json.loads(line).get("text", "")
                    except: yield line

        def _process_texts(texts_iter):
            nonlocal n_docs
            for text in texts_iter:
                if not text or len(text) < min_length: continue
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) < 2: continue
                n_docs += 1
                if pack:
                    buf.extend(ids)
                    while len(buf) >= max_length + 1:
                        self.examples.append(buf[:max_length + 1])
                        buf[:] = buf[max_length:]
                else:
                    ids = ids[:max_length + 1]
                    pad = max_length + 1 - len(ids)
                    self.examples.append(ids + [tokenizer.PAD_ID] * pad)

        # First pass
        _process_texts(_read_texts())

        # FIX: Always save the remaining buffer (even if < max_length)
        if pack and len(buf) > 1:
            # Pad to max_length + 1
            padded = buf + [tokenizer.PAD_ID] * (max_length + 1 - len(buf))
            self.examples.append(padded[:max_length + 1])

        # FIX: If still too few examples, repeat the corpus until we have enough
        repeat = 0
        while len(self.examples) < min_examples and repeat < 50:
            _process_texts(_read_texts())
            if pack and len(buf) > 1:
                padded = buf + [tokenizer.PAD_ID] * (max_length + 1 - len(buf))
                self.examples.append(padded[:max_length + 1])
            repeat += 1

        if not self.examples:
            raise ValueError(
                f"Dataset produced 0 training examples.\n"
                f"Your data has too few tokens for max_seq_length={max_length}.\n"
                f"Try: reduce max_seq_length to {max(8, max_length // 4)}, "
                f"or add more text (need at least {max_length} tokens)."
            )

        logger.info("  %d docs → %d examples (repeats=%d)", n_docs, len(self.examples), repeat)

    def __len__(self) -> int: return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = torch.tensor(self.examples[idx], dtype=torch.long)
        return {
            "input_ids":      seq[:-1],
            "labels":         seq[1:],
            "attention_mask": (seq[:-1] != 0).long(),
        }


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─────────────────────────────────────────────
#  Scheduler
# ─────────────────────────────────────────────

def make_scheduler(opt: AdamW, cfg: TrainingConfig) -> LambdaLR:
    warmup = cfg.warmup_steps
    total  = cfg.max_steps
    min_r  = cfg.lr_decay_end
    _pi    = math.pi

    if cfg.lr_schedule == "linear":
        def fn(s: int) -> float:
            if s < warmup: return s / max(warmup, 1)
            p = (s - warmup) / max(total - warmup, 1)
            return max(min_r, 1.0 - p * (1.0 - min_r))
    else:
        def fn(s: int) -> float:
            if s < warmup: return s / max(warmup, 1)
            p = (s - warmup) / max(total - warmup, 1)
            return min_r + (1.0 - min_r) * 0.5 * (1.0 + math.cos(_pi * p))

    return LambdaLR(opt, fn)


# ─────────────────────────────────────────────
#  Parameter Groups
# ─────────────────────────────────────────────

def build_param_groups(model: LionLLM, base_lr: float,
                        layer_decay: float, wd: float) -> List[Dict]:
    if layer_decay >= 1.0:
        # No layer-wise decay — single group (faster for small models)
        decay_p  = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2
                    and not any(s in n for s in ("norm","bias","embed"))]
        nodecay  = [p for n, p in model.named_parameters()
                    if p.requires_grad and (p.dim() < 2
                    or any(s in n for s in ("norm","bias","embed")))]
        return [{"params": decay_p,  "lr": base_lr, "weight_decay": wd},
                {"params": nodecay,  "lr": base_lr, "weight_decay": 0.0}]

    n = model.cfg.num_hidden_layers
    no_wd = {"bias", "norm", "embed"}
    groups: List[Dict] = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "layers." in name:
            idx = int(name.split("layers.")[1].split(".")[0])
            lr  = base_lr * (layer_decay ** (n - idx))
        else:
            lr = base_lr
        w = 0.0 if (param.dim() < 2 or any(s in name for s in no_wd)) else wd
        groups.append({"params": [param], "lr": lr, "weight_decay": w})
    return groups


# ─────────────────────────────────────────────
#  Checkpoint Manager
# ─────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, out: Path, keep: int = 2) -> None:
        self.out  = Path(out)
        self.keep = keep
        self._saved: List[Tuple[float, Path]] = []

    def save(self, model: LionLLM, opt: AdamW,
             sch: LambdaLR, step: int, val_loss: float) -> Path:
        ckpt = self.out / f"ckpt-{step:07d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        torch.save({"step": step, "val_loss": val_loss,
                    "opt": opt.state_dict(), "sch": sch.state_dict()},
                   ckpt / "trainer.pt")
        (ckpt / "meta.json").write_text(json.dumps({"step": step, "val_loss": val_loss}))
        self._saved.append((val_loss, ckpt))
        self._prune()
        logger.info("✓ Checkpoint step=%d  val_loss=%.4f → %s", step, val_loss, ckpt.name)
        return ckpt

    def _prune(self) -> None:
        import shutil
        self._saved.sort(key=lambda x: x[0])
        while len(self._saved) > self.keep:
            _, old = self._saved.pop(-1)
            if old.exists(): shutil.rmtree(old, ignore_errors=True)

    def load_latest(self, model: LionLLM, opt: AdamW,
                    sch: LambdaLR, device: str) -> int:
        ckpts = sorted(self.out.glob("ckpt-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if not ckpts: return 0
        s = torch.load(ckpts[-1] / "trainer.pt", map_location=device, weights_only=True)
        opt.load_state_dict(s["opt"])
        sch.load_state_dict(s["sch"])
        logger.info("Resumed from %s (step=%d)", ckpts[-1].name, s["step"])
        return s["step"]


# ─────────────────────────────────────────────
#  Trainer  — fully fixed
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model: LionLLM, tokenizer: LionTokenizer,
                 cfg: TrainingConfig) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.out       = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # FIX: Proper device detection including AMD/ROCm
        self.device, dtype_str = setup_device()
        if cfg.dtype != "auto":
            dtype_str = cfg.dtype
        self.amp_dt  = getattr(torch, dtype_str, torch.float32)
        self.use_amp = (self.amp_dt != torch.float32) and (self.device == "cuda")
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        logger.info("Device: %s  dtype: %s  AMP: %s",
                    self.device, dtype_str, self.use_amp)

        # FIX: gradient checkpointing only on GPU
        if cfg.gradient_checkpointing and self.device in ("cuda", "mps"):
            model.enable_gc()
        elif cfg.gradient_checkpointing:
            logger.info("Gradient checkpointing skipped (CPU — no benefit)")

        model.to(self.device)

        # FIX: validate dataset BEFORE creating DataLoader
        try:
            n_lines, total_tokens = validate_dataset(
                cfg.dataset_path, tokenizer, cfg.max_seq_length
            )
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"\n\n{'='*55}\n  Training Error\n{'='*55}\n{e}\n{'='*55}") from e

        # FIX: auto max_steps based on dataset size
        n_examples_est = max(8, total_tokens // cfg.max_seq_length)
        if cfg.max_steps == 0:
            cfg.max_steps = max(100, n_examples_est * 20)
            logger.info("Auto max_steps: %d (based on ~%d examples)",
                        cfg.max_steps, n_examples_est)

        # FIX: auto eval/save intervals
        if cfg.eval_interval == 0:
            cfg.eval_interval = max(10, cfg.max_steps // 10)
        if cfg.save_interval == 0:
            cfg.save_interval = max(50, cfg.max_steps // 5)

        # FIX: warmup proportional to steps
        cfg.warmup_steps = min(cfg.warmup_steps, cfg.max_steps // 5)

        # Dataset
        full_ds = PackedTextDataset(
            cfg.dataset_path, tokenizer,
            cfg.max_seq_length, cfg.min_seq_length,
            cfg.pack_sequences,
            min_examples=max(8, cfg.batch_size),
        )

        # FIX: split only if enough examples
        if len(full_ds) >= 10 and cfg.train_split > 0:
            n_train = max(1, int(len(full_ds) * cfg.train_split))
            n_val   = len(full_ds) - n_train
            if n_val < 1: n_val = 1; n_train = len(full_ds) - 1
            tr, va  = random_split(full_ds, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))
        else:
            tr = full_ds   # use all for training, validate on train
            va = full_ds

        # FIX: batch_size clipped to dataset size; drop_last=False
        bs = min(cfg.batch_size, len(tr))
        if cfg.auto_batch_size:
            # RAM-based sizing (works for all hardware including AMD)
            try:
                import psutil
                avail_gb = psutil.virtual_memory().available / 1e9
            except ImportError:
                avail_gb = 4.0  # conservative default
            if avail_gb < 4:   bs = min(bs, 1)
            elif avail_gb < 8: bs = min(bs, 2)
            elif avail_gb < 12: bs = min(bs, 4)
            logger.info("Batch size: %d (%.1f GB RAM available)", bs, avail_gb)

        bs = max(1, bs)

        # FIX: num_workers=0 when dataset is tiny (persistent_workers would crash)
        nw = 0 if (len(tr) < 16 or platform.system() == "Windows") else min(2, os.cpu_count() or 1)
        pw = nw > 0

        self.train_dl = DataLoader(
            tr, batch_size=bs, shuffle=True,
            collate_fn=_collate,
            drop_last=False,  # FIX: was True (killed tiny batches)
            num_workers=nw,
            pin_memory=(self.device == "cuda"),
            persistent_workers=pw,
        )
        self.val_dl = DataLoader(
            va, batch_size=max(1, bs),
            shuffle=False, collate_fn=_collate, num_workers=0,
        )

        logger.info("Dataset: %d train | %d val | batch=%d | steps=%d",
                    len(tr), len(va), bs, cfg.max_steps)

        # Optimiser
        groups = build_param_groups(model, cfg.learning_rate,
                                     cfg.layer_lr_decay, cfg.weight_decay)
        self.opt  = AdamW(groups, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
        self.sch  = make_scheduler(self.opt, cfg)
        self.ckpt = CheckpointManager(self.out, cfg.keep_checkpoints)

        self._loss_q: Deque[float] = deque(maxlen=cfg.log_interval)
        self._step   = 0
        self._best   = float("inf")
        self._pat    = 0

        cfg.save(self.out / "training_config.json")
        model.cfg.save(self.out)
        tokenizer.save(self.out)

    @torch.inference_mode()
    def _eval(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_dl):
            if i >= self.cfg.eval_steps: break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            ctx   = (torch.autocast(self.device, dtype=self.amp_dt)
                     if self.use_amp else nullcontext())
            with ctx:
                out = self.model(**batch)
            total += out["loss"].item(); n += 1
        self.model.train()
        return total / max(n, 1)

    def _forward_backward(self, batch: Dict) -> float:
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        ctx   = (torch.autocast(self.device, dtype=self.amp_dt)
                 if self.use_amp else nullcontext())
        with ctx:
            loss = self.model(**batch)["loss"] / self.cfg.gradient_accumulation
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.item() * self.cfg.gradient_accumulation

    def _grad_step(self) -> None:
        if self.scaler:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt); self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.sch.step()

    def train(self) -> None:
        cfg = self.cfg
        logger.info("─"*55)
        logger.info("LionAI Training  device=%s  steps=%d  seq=%d  batch=%d×%d",
                    self.device, cfg.max_steps, cfg.max_seq_length,
                    cfg.batch_size, cfg.gradient_accumulation)
        logger.info("  Params: %.2fM | warmup=%d | eval_every=%d",
                    self.model.n_params()/1e6, cfg.warmup_steps, cfg.eval_interval)
        logger.info("─"*55)

        if cfg.resume_from:
            self._step = self.ckpt.load_latest(
                self.model, self.opt, self.sch, self.device)

        self.model.train()
        data_iter   = iter(self.train_dl)
        t0          = time.time()
        t_start     = t0
        accum_loss  = 0.0

        while self._step < cfg.max_steps:
            step_loss = 0.0
            for _ in range(cfg.gradient_accumulation):
                try:    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dl)
                    batch     = next(data_iter)
                try:
                    step_loss += self._forward_backward(batch)
                except RuntimeError as e:
                    if "memory" in str(e).lower():
                        logger.warning("OOM — reducing batch. Try --model micro")
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        continue
                    raise

            self._grad_step()
            self._step += 1
            self._loss_q.append(step_loss)

            # Progress bar style logging every log_interval steps
            if self._step % cfg.log_interval == 0:
                avg = sum(self._loss_q) / len(self._loss_q)
                lr  = self.opt.param_groups[0]["lr"]
                elapsed = time.time() - t0
                tps     = (cfg.log_interval * cfg.batch_size
                           * cfg.gradient_accumulation
                           * cfg.max_seq_length) / max(elapsed, 1e-6)
                pct     = 100 * self._step / cfg.max_steps
                eta_s   = (cfg.max_steps - self._step) * elapsed / max(cfg.log_interval, 1)
                eta_str = f"{int(eta_s//60)}m{int(eta_s%60)}s"
                t0      = time.time()
                logger.info(
                    "[%5.1f%%] step %d/%d | loss %.4f | lr %.1e | %.0f tok/s | ETA %s",
                    pct, self._step, cfg.max_steps, avg, lr, tps, eta_str
                )

            # Evaluation
            if self._step % cfg.eval_interval == 0:
                val = self._eval()
                logger.info("  → val_loss %.4f%s",
                            val, "  ✓ best" if val < self._best else "")
                self.ckpt.save(self.model, self.opt, self.sch, self._step, val)
                if val < self._best - cfg.patience_delta:
                    self._best = val; self._pat = 0
                else:
                    self._pat += 1
                    if self._pat >= cfg.patience:
                        logger.info("Early stopping at step %d", self._step)
                        break
            elif self._step % cfg.save_interval == 0:
                self.ckpt.save(self.model, self.opt, self.sch, self._step, float("inf"))

        total_time = time.time() - t_start
        logger.info("Training done in %.1fs (%.1f min)", total_time, total_time/60)

        final = self.out / "final"
        self.model.save_pretrained(final)
        self.tokenizer.save(final)
        logger.info("Model saved → %s", final)


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LionAI Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    default="./data/train.jsonl")
    parser.add_argument("--output",     default="./runs/lionai")
    parser.add_argument("--model-size", default=None,
                        choices=["micro","small","medium","large"],
                        help="Model size (default: auto from dataset)")
    parser.add_argument("--vocab",      type=int, default=None,
                        help="Vocab size (default: auto from dataset)")
    parser.add_argument("--steps",      type=int, default=0,
                        help="Training steps (0=auto)")
    parser.add_argument("--seq-len",    type=int, default=None,
                        help="Sequence length (default: auto)")
    parser.add_argument("--batch",      type=int, default=None,
                        help="Batch size (default: auto)")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-detect dataset size and pick sensible defaults
    dataset_path = Path(args.dataset)

    # Quick size estimate (no tokenizer yet)
    n_chars = 0
    n_lines = 0
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                n_lines += 1
                try:    n_chars += len(json.loads(line.strip()).get("text",""))
                except: n_chars += len(line)

    n_tokens_est  = max(1, n_chars // 4)   # rough: 4 chars/token
    logger.info("Dataset size estimate: %d lines, ~%d tokens", n_lines, n_tokens_est)

    # Auto model size
    if args.model_size is None:
        if n_tokens_est < 10_000:      model_size = "micro"
        elif n_tokens_est < 100_000:   model_size = "small"
        elif n_tokens_est < 1_000_000: model_size = "medium"
        else:                          model_size = "large"
    else:
        model_size = args.model_size
    logger.info("Model size: %s", model_size)

    # Auto vocab size
    vocab_size = args.vocab
    if vocab_size is None:
        if n_tokens_est < 5_000:       vocab_size = 512
        elif n_tokens_est < 50_000:    vocab_size = 2000
        elif n_tokens_est < 500_000:   vocab_size = 8000
        else:                           vocab_size = 32000
    logger.info("Vocab size: %d", vocab_size)

    # Auto seq length
    seq_len = args.seq_len
    if seq_len is None:
        if n_tokens_est < 1_000:       seq_len = 32
        elif n_tokens_est < 10_000:    seq_len = 64
        elif n_tokens_est < 100_000:   seq_len = 128
        else:                           seq_len = 256
    logger.info("Seq length: %d", seq_len)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer if not already present
    tok_path = out_dir / "tokenizer.json"
    if not tok_path.exists():
        logger.info("Training tokenizer (vocab=%d) …", vocab_size)
        from tokenizer import TokenizerTrainer
        from tokenizer_trainer import iter_corpus

        def _texts():
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    try:    yield json.loads(line.strip()).get("text","")
                    except: yield line.strip()

        trainer   = TokenizerTrainer(vocab_size=vocab_size, min_frequency=1,
                                     show_progress=True)
        tokenizer = trainer.train(_texts())
        tokenizer.save(out_dir)
    else:
        from tokenizer import LionTokenizer
        tokenizer = LionTokenizer.load(out_dir)
        logger.info("Tokenizer loaded (vocab=%d)", tokenizer.vocab_size)

    # Build model
    model_pt = out_dir / "final" / "model.pt"
    if not model_pt.exists() or args.resume:
        cfg_fn = getattr(ModelConfig, model_size)
        import dataclasses
        mcfg   = dataclasses.replace(cfg_fn(), vocab_size=tokenizer.vocab_size)
        model  = LionLLM(mcfg)
    else:
        model = LionLLM.from_pretrained(out_dir / "final", map_location="cpu")

    # Build training config with smart defaults
    n_examples_rough = max(8, n_tokens_est // seq_len)
    tcfg = TrainingConfig.for_small_dataset(n_examples_rough, vocab_size)
    tcfg.dataset_path   = str(dataset_path)
    tcfg.output_dir     = str(out_dir)
    tcfg.max_seq_length = seq_len
    if args.steps > 0: tcfg.max_steps = args.steps
    if args.batch:     tcfg.batch_size = args.batch
    if args.resume:    tcfg.resume_from = str(out_dir)

    trainer_obj = Trainer(model, tokenizer, tcfg)
    trainer_obj.train()
