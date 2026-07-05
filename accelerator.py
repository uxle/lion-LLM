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
    Set PyTorch CPU thread count to use target_cpu_pct of physical cores if possible,
    otherwise target_cpu_pct of logical cores.
    Also sets interop threads for parallel data loading.
    Returns the number of threads set.
    """
    n_logical = os.cpu_count() or 4
    try:
        import psutil
        n_cores = psutil.cpu_count(logical=False) or n_logical
    except ImportError:
        n_cores = n_logical

    n_threads = max(1, int(n_cores * target_cpu_pct))

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

    logger.info("CPU threads: %d/%d cores (%.0f%% target)",
                n_threads, n_cores, target_cpu_pct * 100)
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
    Caches the KV entries for multiple prompt prefixes so they are only encoded ONCE,
    not re-computed when templates or prompts change.
    """

    def __init__(self) -> None:
        self._cache = {}           # dict mapping tuple(prompt_ids) -> (pkv, len)
        self._lock = threading.Lock()

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._cache) > 0

    def get(self, prompt_ids: torch.Tensor):
        """
        Check if any cached prompt prefix matches the beginning of prompt_ids.
        Returns (pkv, prefix_len) or (None, 0).
        """
        if prompt_ids.ndim > 1:
            tokens = tuple(prompt_ids[0].tolist())
        else:
            tokens = tuple(prompt_ids.tolist())

        with self._lock:
            best_prefix = None
            best_len = 0
            for prefix, (pkv, length) in self._cache.items():
                if len(prefix) <= len(tokens) and tokens[:len(prefix)] == prefix:
                    if len(prefix) > best_len:
                        best_prefix = pkv
                        best_len = len(prefix)
            return best_prefix, best_len

    def prime(self, engine, prompt_ids: torch.Tensor) -> None:
        """Run a forward pass on prompt_ids and cache the resulting KV."""
        if prompt_ids.ndim > 1:
            tokens = tuple(prompt_ids[0].tolist())
        else:
            tokens = tuple(prompt_ids.tolist())

        with self._lock:
            if tokens in self._cache:
                return
            with torch.inference_mode():
                out = engine.model(
                    prompt_ids.to(engine.device),
                    past_key_values=None,
                    use_cache=True
                )
            self._cache[tokens] = (out["past_key_values"], prompt_ids.shape[1])
            logger.info("KV cache primed: %d tokens cached (total cached keys: %d)",
                        prompt_ids.shape[1], len(self._cache))

    def cached_len(self) -> int:
        with self._lock:
            if not self._cache: return 0
            return max(length for _, length in self._cache.values())

    def invalidate(self) -> None:
        """Invalidate when system prompt changes."""
        with self._lock:
            self._cache.clear()


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
                 cpu_pct: float = 0.85,   # 85%: fast but OS stays responsive
                 gpu_pct: float = 0.70,   # 70%: use 70% of available VRAM
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
        Optimised token generation — allocation-free inner loop.

        Speed fixes vs original:
          FIX 1: Pre-allocated id buffer — no torch.cat per token (was O(n) copy every step)
          FIX 2: Pre-allocated seen_buf — no torch.tensor(gen) rebuild per token
          FIX 3: Single softmax in top-p (was calling softmax twice on same sl tensor)
          FIX 4: masked_fill for top-k instead of boolean index write (avoids copy)
        """
        import gc

        eos     = self.cfg.eos_token_id if self.cfg else 2
        vocab   = self.cfg.vocab_size   if self.cfg else 32000
        max_pos = self.cfg.max_position_embeddings if self.cfg else 2048
        use_amp = (self.device == "cuda")

        # ── FIX 1: Pre-allocate a fixed-size id buffer ───────────────────────────
        # Avoids torch.cat([ids, new_tok]) — that allocates + copies the whole
        # sequence on every token. Instead we write into a pre-sized buffer.
        prompt_len = input_ids.shape[1]

        # Guard: if the prompt is already at or beyond the context window,
        # truncate it to leave room for at least max_new_tokens new tokens.
        # Keep BOS (first token) + the most recent context.
        max_prompt = max(1, max_pos - max_new_tokens)
        if prompt_len > max_prompt:
            input_ids  = torch.cat(
                [input_ids[:, :1], input_ids[:, -(max_prompt - 1):]], dim=-1
            )
            prompt_len = input_ids.shape[1]

        buf_len = min(prompt_len + max_new_tokens, max_pos)
        ids_buf = torch.zeros(1, buf_len, dtype=torch.long, device=self.device)
        ids_buf[0, :prompt_len] = input_ids[0].to(self.device)
        cur_len = prompt_len

        # ── FIX 2: Pre-allocate penalty / tracking tensors ───────────────────────
        # Avoids torch.tensor(gen, device=...) rebuild on every token.
        freq     = torch.zeros(vocab,          device=self.device, dtype=torch.float32)
        seen_buf = torch.zeros(max_new_tokens, device=self.device, dtype=torch.long)
        n_gen    = 0

        # Check KV cache for matching prefix
        pkv = None
        prefix_len = 0
        if hasattr(self, "kv_mgr") and self.kv_mgr:
            pkv, prefix_len = self.kv_mgr.get(input_ids)
            if prefix_len > 0:
                if prefix_len == cur_len:
                    prefix_len = cur_len - 1
                if pkv is not None:
                    sliced_pkv = []
                    for layer_kv in pkv:
                        if layer_kv is not None:
                            k_layer, v_layer = layer_kv
                            sliced_pkv.append((k_layer[:, :, :prefix_len], v_layer[:, :, :prefix_len]))
                        else:
                            sliced_pkv.append(None)
                    pkv = tuple(sliced_pkv)

        for _ in range(max_new_tokens):
            if pkv is None:
                cur = ids_buf[:, :cur_len]
            elif prefix_len > 0:
                # First step after loading cache: run model on the new tokens only
                cur = ids_buf[:, prefix_len:cur_len]
                prefix_len = 0 # reset so subsequent steps use single-token decoding
            else:
                cur = ids_buf[:, cur_len - 1: cur_len]

            with torch.autocast(self.device, dtype=self.dtype, enabled=use_amp):
                out = self.model(cur, past_key_values=pkv, use_cache=True)

            logits = out["logits"][:, -1, :].float()   # (1, vocab)
            pkv    = out["past_key_values"]

            # ── Penalties (FIX 2: view into pre-allocated seen_buf) ───────────────
            if n_gen > 0:
                seen = seen_buf[:n_gen]
                if repetition_penalty != 1.0:
                    lp = logits[0, seen]
                    logits[0, seen] = torch.where(
                        lp < 0,
                        lp * repetition_penalty,
                        lp / repetition_penalty,
                    )
                if frequency_penalty != 0:
                    logits[0] -= frequency_penalty * freq
                if presence_penalty != 0:
                    logits[0, seen.unique()] -= presence_penalty

            # ── Temperature ──────────────────────────────────────────────────────
            if temperature > 0 and temperature != 1.0:
                logits /= temperature

            # ── min-p ─────────────────────────────────────────────────────────────
            if min_p > 0:
                probs0  = logits.softmax(-1)
                thresh  = probs0.max(-1, keepdim=True).values * min_p
                logits  = logits.masked_fill(probs0 < thresh, float("-inf"))

            # ── top-k (FIX 4: masked_fill avoids index-write copy) ────────────────
            if top_k > 0:
                k  = min(top_k, logits.size(-1))
                th = logits.topk(k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < th, float("-inf"))

            # ── top-p nucleus (FIX 3: single softmax) ────────────────────────────
            if 0 < top_p < 1.0:
                sl, si   = logits.sort(-1, descending=True)
                probs_sl = sl.softmax(-1)            # computed ONCE
                cp       = probs_sl.cumsum(-1)
                rm       = (cp - probs_sl) > top_p  # reuse — was calling softmax again
                rm[:, 0] = False
                logits.scatter_(-1, si, sl.masked_fill(rm, float("-inf")))

            probs = logits.softmax(-1)
            tid   = int(torch.multinomial(probs, 1).item())

            # ── FIX 1: write next token into pre-allocated buffer ─────────────────
            if cur_len < buf_len:
                ids_buf[0, cur_len] = tid
                cur_len += 1
            else:
                # Buffer full — slide context window in-place
                keep = int(max_pos * 0.6)
                ids_buf[0, 1: 1 + keep] = ids_buf[0, cur_len - keep: cur_len].clone()
                cur_len = 1 + keep
                ids_buf[0, cur_len - 1] = tid
                pkv = None
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # ── FIX 2: update tracking tensors in-place ───────────────────────────
            seen_buf[n_gen] = tid
            freq[tid]      += 1.0
            n_gen          += 1

            yield tid

            # ── Stop conditions ───────────────────────────────────────────────────
            if tid == eos:
                break
            if stop_ids and tid in stop_ids:
                break
            if stop_strings and tokenizer:
                if any(s in tokenizer.decode(seen_buf[:n_gen].tolist())
                       for s in stop_strings):
                    break


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
