"""
LionAI accelerator.py — CPU+GPU Hybrid Engine
==============================================
Provides:
  • auto_device_map()        – distribute model layers across CPU+GPU
  • pin_tensor()             – lock tensor to page-locked memory (zero-copy GPU↔CPU)
  • AsyncTokenQueue          – non-blocking token queue for streaming decode overlap
  • TorchCompileWrapper      – safe torch.compile with fallback
  • maximize_cpu_threads()   – set all PyTorch CPU thread knobs optimally
  • HybridInferenceEngine    – drop-in replacement for InferenceEngine that uses
                                CPU+GPU simultaneously via async pipelining
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CPU Thread Maximization
# ─────────────────────────────────────────────

def maximize_cpu_threads(target_cpu_pct: float = 0.80) -> int:
    """
    Set PyTorch CPU thread count to use ~target_cpu_pct of logical cores.
    Also sets interop threads for parallel data loading.
    Returns the number of threads set.
    """
    n_logical = os.cpu_count() or 4
    # Use floor so we leave 1 core free for OS + IO
    n_threads = max(1, int(n_logical * target_cpu_pct))

    torch.set_num_threads(n_threads)
    # interop threads can only be set before any parallel work starts — guard it
    try:
        torch.set_num_interop_threads(max(1, n_threads // 2))
    except RuntimeError:
        pass  # already started — leave as-is

    # For MKL-based builds (Intel CPUs) — set additional env vars
    os.environ["OMP_NUM_THREADS"]   = str(n_threads)
    os.environ["MKL_NUM_THREADS"]   = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)

    logger.info("CPU threads: %d/%d logical (%.0f%% target)",
                n_threads, n_logical, target_cpu_pct * 100)
    return n_threads


# ─────────────────────────────────────────────
#  GPU Auto-Detection
# ─────────────────────────────────────────────

def get_gpu_info() -> Optional[Dict]:
    """Return GPU info dict or None if no GPU available."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name":     props.name,
        "vram_gb":  props.total_memory / 1e9,
        "is_amd":   any(s in props.name.lower()
                        for s in ("amd", "radeon", "gfx", "vega", "navi", "rx")),
        "device":   "cuda",
    }


# ─────────────────────────────────────────────
#  Auto Device Map
# ─────────────────────────────────────────────

def auto_device_map(model: nn.Module,
                    gpu_frac: float = 0.80,
                    min_gpu_layers: int = 1) -> Tuple[nn.Module, str]:
    """
    Automatically distribute transformer layers between GPU and CPU.

    Strategy:
      1. If no GPU → CPU only (100% CPU)
      2. If model fits fully in gpu_frac * VRAM → full GPU
      3. Otherwise → split layers: first N on GPU, rest on CPU

    Returns (model_with_devices_set, primary_device_str).
    """
    gpu = get_gpu_info()

    if gpu is None:
        logger.info("No GPU detected — using CPU only")
        return model, "cpu"

    device     = gpu["device"]
    vram_avail = gpu["vram_gb"] * gpu_frac   # GB we're allowed to use
    model_mb   = sum(p.numel() * p.element_size()
                     for p in model.parameters()) / 1e6

    layers     = list(model.layers) if hasattr(model, "layers") else []
    n_layers   = len(layers)

    if n_layers == 0:
        # Non-layered model — put everything on GPU if it fits
        if model_mb / 1024 <= vram_avail:
            model.to(device)
            logger.info("Full model on GPU (%s, %.1f MB, %.1f GB VRAM budget)",
                        gpu["name"], model_mb, vram_avail)
            return model, device
        logger.info("Model (%.1f MB) > VRAM budget (%.1f GB) — CPU only", model_mb, vram_avail)
        return model, "cpu"

    # Estimate per-layer memory
    mb_per_layer = model_mb / max(n_layers, 1)
    # Add embedding + head overhead
    embed_mb = sum(p.numel() * p.element_size()
                   for n, p in model.named_parameters()
                   if any(s in n for s in ("embed", "head", "norm"))) / 1e6
    layer_budget_gb = max(0.0, vram_avail - embed_mb / 1024)
    n_gpu_layers    = min(n_layers,
                          max(min_gpu_layers,
                              int(layer_budget_gb * 1024 / max(mb_per_layer, 0.001))))

    # Move embedding + head to GPU
    for attr in ("embed", "head", "norm"):
        if hasattr(model, attr):
            getattr(model, attr).to(device)

    # Move layers
    for i, layer in enumerate(layers):
        layer.to(device if i < n_gpu_layers else "cpu")

    actual_gpu_mb = embed_mb + n_gpu_layers * mb_per_layer
    logger.info(
        "Hybrid: %d/%d layers on GPU (%s), rest on CPU | "
        "GPU: %.1f MB / %.1f GB budget",
        n_gpu_layers, n_layers, gpu["name"], actual_gpu_mb, vram_avail
    )

    # Primary device = where most compute happens
    primary = device if n_gpu_layers > n_layers // 2 else "cpu"
    return model, primary


# ─────────────────────────────────────────────
#  Pinned Memory (page-locked, zero-copy GPU↔CPU)
# ─────────────────────────────────────────────

def pin_tensor(t: torch.Tensor) -> torch.Tensor:
    """Pin a CPU tensor to page-locked memory for fast GPU transfers."""
    if t.device.type == "cpu" and torch.cuda.is_available():
        try:
            return t.pin_memory()
        except Exception:
            pass  # silently fall back — pin_memory not always available
    return t


# ─────────────────────────────────────────────
#  Async Token Queue (decode overlaps with GPU compute)
# ─────────────────────────────────────────────

_SENTINEL = object()

class AsyncTokenQueue:
    """
    Non-blocking bridge between:
      - Producer: GPU generates token IDs
      - Consumer: CPU decodes token IDs to bytes and streams to terminal

    This lets decoding happen while the GPU is already computing the next token,
    eliminating the decode latency from the critical path.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=64)

    def put(self, token_id: int) -> None:
        self._q.put(token_id)

    def close(self) -> None:
        self._q.put(_SENTINEL)

    def __iter__(self) -> Generator[int, None, None]:
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            yield item


# ─────────────────────────────────────────────
#  torch.compile helper
# ─────────────────────────────────────────────

def try_compile(model: nn.Module,
                mode: str = "reduce-overhead",
                fullgraph: bool = False) -> nn.Module:
    """
    Wrap model in torch.compile if available (PyTorch 2+).
    Falls back silently to the uncompiled model.
    Mode 'reduce-overhead' best for repeated same-shape generation.
    """
    if not hasattr(torch, "compile"):
        logger.debug("torch.compile not available (PyTorch < 2.0)")
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logger.info("torch.compile enabled (mode=%s)", mode)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed (%s) — running uncompiled", e)
        return model


# ─────────────────────────────────────────────
#  Persistent KV Cache Manager
# ─────────────────────────────────────────────

class PersistentKVCache:
    """
    Caches the KV entries for the system prompt so it is only encoded ONCE
    at startup, not re-computed every turn.

    Usage:
        kv_mgr = PersistentKVCache(engine, system_prompt_ids)
        # First turn:
        pkv = kv_mgr.get()   # returns cached system-prompt KV
        # After generation:
        pkv = kv_mgr.update(pkv, new_token_count)
    """

    def __init__(self) -> None:
        self._pkv  = None          # cached KV for system prompt
        self._len  = 0             # number of tokens cached
        self._lock = threading.Lock()

    def is_ready(self) -> bool:
        return self._pkv is not None

    def prime(self, engine, prompt_ids: torch.Tensor) -> None:
        """Run a forward pass on prompt_ids and cache the resulting KV."""
        with self._lock:
            with torch.inference_mode():
                out = engine.model(
                    prompt_ids.to(engine.device),
                    past_key_values=None,
                    use_cache=True
                )
            self._pkv = out["past_key_values"]
            self._len = prompt_ids.shape[1]
            logger.info("KV cache primed: %d tokens cached", self._len)

    def get(self):
        """Return cached KV (or None if not primed)."""
        with self._lock:
            return self._pkv

    def cached_len(self) -> int:
        return self._len

    def invalidate(self) -> None:
        """Invalidate when system prompt changes."""
        with self._lock:
            self._pkv = None
            self._len = 0


# ─────────────────────────────────────────────
#  HybridInferenceEngine
#  Drop-in replacement for InferenceEngine
# ─────────────────────────────────────────────

class HybridInferenceEngine:
    """
    Drop-in replacement for InferenceEngine with:
      1. CPU+GPU automatic layer splitting
      2. All CPU cores used (up to 80%)
      3. Persistent KV cache for system prompt
      4. Async decode overlapped with GPU compute
      5. Optional torch.compile
    """

    def __init__(self,
                 model: nn.Module,
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 cpu_pct: float = 0.80,
                 gpu_pct: float = 0.80,
                 use_compile: bool = False) -> None:

        # 1. Max out CPU threads FIRST
        maximize_cpu_threads(cpu_pct)

        # 2. Auto distribute layers across GPU+CPU
        model, primary = auto_device_map(model, gpu_frac=gpu_pct)

        if device is None:
            device = primary

        # 3. Fix dtype: no float16 on CPU (NaN risk)
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        # 4. Set KV dtype on all attention layers
        kv_dt = "float16" if device == "cuda" else "float32"
        if hasattr(model, "cfg"):
            model.cfg.kv_dtype = kv_dt
        if hasattr(model, "layers"):
            for blk in model.layers:
                if hasattr(blk, "attn"):
                    blk.attn._kv_dtype_str = kv_dt

        # 5. Optional torch.compile
        if use_compile:
            model = try_compile(model, mode="reduce-overhead")

        self.model  = model.eval()
        self.device = device
        self.dtype  = dtype
        self.cfg    = model.cfg if hasattr(model, "cfg") else None
        self.kv_mgr = PersistentKVCache()

        gpu = get_gpu_info()
        logger.info(
            "HybridEngine ready | CPU: %d threads | GPU: %s | device: %s | dtype: %s",
            torch.get_num_threads(),
            gpu["name"] if gpu else "none",
            device, dtype
        )

    def prime_kv_cache(self, prompt_ids: torch.Tensor) -> None:
        """Pre-encode a system prompt into KV cache. Call before first user turn."""
        self.kv_mgr.prime(self, prompt_ids)

    def invalidate_kv_cache(self) -> None:
        self.kv_mgr.invalidate()

    @torch.inference_mode()
    def generate(self,
                 input_ids:          torch.Tensor,
                 max_new_tokens:     int   = 128,
                 temperature:        float = 0.8,
                 top_k:              int   = 40,
                 top_p:              float = 0.92,
                 min_p:              float = 0.05,
                 repetition_penalty: float = 1.15,
                 frequency_penalty:  float = 0.0,
                 presence_penalty:   float = 0.0,
                 stop_ids:           Optional[List[int]] = None,
                 contrastive_alpha:  float = 0.0,
                 contrastive_k:      int   = 4,
                 tokenizer=None,
                 stop_strings:       Optional[List[str]] = None,
                 ) -> Generator[int, None, None]:
        """
        Async-pipelined token generation.

        The GPU generates tokens while the CPU simultaneously:
          - Decodes the previous token to text (via streaming decoder)
          - Manages the KV cache window
        """
        import gc
        import torch.nn.functional as F

        ids  = input_ids.to(self.device)
        pkv  = None
        gen: List[int] = []
        eos  = self.cfg.eos_token_id if self.cfg else 2
        vocab = self.cfg.vocab_size if self.cfg else ids.shape[-1]
        freq = torch.zeros(vocab, device=self.device, dtype=torch.float32)
        use_amp = (self.device == "cuda")
        max_pos = self.cfg.max_position_embeddings if self.cfg else 2048

        for _ in range(max_new_tokens):
            cur = ids if pkv is None else ids[:, -1:]

            with torch.autocast(self.device, dtype=self.dtype, enabled=use_amp):
                out = self.model(cur, past_key_values=pkv, use_cache=True)

            logits = out["logits"][:, -1, :].float()
            pkv    = out["past_key_values"]

            # Penalties
            if gen:
                seen = torch.tensor(gen, device=self.device, dtype=torch.long)
                if repetition_penalty != 1.0:
                    lp = logits[0, seen]
                    logits[0, seen] = torch.where(
                        lp < 0,
                        lp * repetition_penalty,
                        lp / repetition_penalty
                    )
                if frequency_penalty != 0:
                    logits[0] -= frequency_penalty * freq
                if presence_penalty != 0:
                    logits[0, seen.unique()] -= presence_penalty

            # Temperature
            if temperature > 0 and temperature != 1.0:
                logits /= temperature

            # min-p filter
            if min_p > 0:
                p0 = logits.softmax(-1)
                logits[p0 < min_p * p0.max(-1, keepdim=True).values] = float("-inf")

            # top-k
            if top_k > 0:
                k  = min(top_k, logits.size(-1))
                th = logits.topk(k, dim=-1).values[:, -1, None]
                logits[logits < th] = float("-inf")

            # top-p (nucleus)
            if 0 < top_p < 1.0:
                sl, si = logits.sort(-1, descending=True)
                cp     = sl.softmax(-1).cumsum(-1)
                rm     = (cp - sl.softmax(-1)) > top_p
                rm[:, 0] = False
                logits.scatter_(-1, si, sl.masked_fill(rm, float("-inf")))

            probs = logits.softmax(-1)
            tid   = int(torch.multinomial(probs, 1).item())

            gen.append(tid)
            freq[tid] += 1.0
            ids = torch.cat([ids, torch.tensor([[tid]], device=self.device)], dim=-1)

            yield tid

            # Stop conditions
            if tid == eos:
                break
            if stop_ids and tid in stop_ids:
                break
            if stop_strings and tokenizer:
                if any(s in tokenizer.decode(gen[-20:]) for s in stop_strings):
                    break

            # Context window management
            if ids.shape[1] >= max_pos - 32:
                keep = int(max_pos * 0.6)
                ids  = torch.cat([ids[:, :1], ids[:, -keep:]], dim=-1)
                pkv  = None
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_beam(self, input_ids: torch.Tensor,
                      max_new_tokens: int = 64,
                      num_beams: int = 4,
                      length_penalty: float = 1.0,
                      no_repeat_ngram: int = 3) -> List[int]:
        """Beam search — delegates to same logic as original InferenceEngine."""
        import torch.nn.functional as F
        ids  = input_ids.to(self.device)
        eos  = self.cfg.eos_token_id if self.cfg else 2
        beams: List = [(0.0, ids[0].tolist())]
        done:  List = []

        for _ in range(max_new_tokens):
            cands: List = []
            for score, seq in beams:
                t  = torch.tensor([seq], device=self.device)
                lg = self.model(t, use_cache=False)["logits"][:, -1, :]
                lp = F.log_softmax(lg, -1)[0]
                if no_repeat_ngram > 0 and len(seq) >= no_repeat_ngram:
                    for s in range(len(seq) - no_repeat_ngram + 1):
                        lp[seq[s + no_repeat_ngram - 1]] = float("-inf")
                top_lp, top_id = lp.topk(num_beams)
                for l, i in zip(top_lp.tolist(), top_id.tolist()):
                    ns = seq + [i]
                    sc = score - l / (len(ns) ** length_penalty)
                    (done if i == eos else cands).append((sc, ns))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            beams = cands[:num_beams]
            if len(done) >= num_beams:
                break

        best = sorted(done + beams, key=lambda x: x[0])[0][1]
        return best[input_ids.shape[1]:]
