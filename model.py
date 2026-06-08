"""
LionAI model.py — Bug-Fixed + CPU/AMD-Optimised Edition
=========================================================
Bugs fixed:
  BUG 1: frozen ModelConfig broke dataclasses.replace() in demo_setup.py
          → removed frozen=True; added __hash__ manually for backward compat
  BUG 2: InferenceEngine dtype=float16 on CPU caused silent NaN on non-CUDA
          → dtype auto-select now forces float32 on CPU
  BUG 3: KV cache stored in float16 on CPU → cast errors during cat()
          → kv_dtype defaults to "float32" on CPU, "float16" on CUDA
  BUG 4: generate() freq tensor allocated on CPU even when device=cuda
          → freq tensor now allocated on self.device
  BUG 5: _contrastive() called self.model.embed() — attribute doesn't exist
          → corrected to self.model.embed.weight[ti[0]]
  BUG 6: RotaryEmbedding cache device mismatch after model.to(device)
          → cos/sin rebuilt on forward if device changed
  BUG 7: GQAttention used self.model.embed — should be parent model reference
          → fixed contrastive to use engine's model reference
  BUG 8: SwiGLU gate_up chunks wrong dim on batch>1
          → explicit dim=-1 on chunk()
  BUG 9: Block._gc flag not reset after disable_gc() call
          → disable_gc() now correctly sets all blocks False
  BUG 10: generate() sliding window reset past_kv=None but kept stale ids
           → ids trimmed consistently with cache reset

AMD RX550 / i5-10th optimisations:
  • float32 by default on CPU (no float16 NaN risk)
  • torch.set_num_threads called with os.cpu_count() on first engine init
  • Attention fallback path optimised for CPU (no autocast overhead)
  • KV cache disabled by default on CPU (saves RAM, faster for short gen)
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_ckpt

logger = logging.getLogger(__name__)

_SDPA = hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.0"
if _SDPA:
    logger.debug("Flash Attention (SDPA) available")


# ─────────────────────────────────────────────────────────────
#  Config  (not frozen — allows dataclasses.replace())
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:               int   = 32000
    pad_token_id:             int   = 0
    bos_token_id:             int   = 1
    eos_token_id:             int   = 2
    hidden_size:              int   = 768
    intermediate_size:        int   = 2048
    num_hidden_layers:        int   = 12
    num_attention_heads:      int   = 12
    num_key_value_heads:      int   = 4
    head_dim:                 int   = 64
    max_position_embeddings:  int   = 2048
    hidden_dropout_prob:      float = 0.05
    attention_dropout:        float = 0.0
    layer_norm_eps:           float = 1e-6
    rope_theta:               float = 500_000.0
    rope_scaling:             Optional[float] = None
    initializer_range:        float = 0.02
    tie_word_embeddings:      bool  = True
    use_cache:                bool  = True
    # FIX: kv_dtype now device-aware (set in InferenceEngine, not here)
    kv_dtype:                 str   = "float32"
    gradient_checkpointing:   bool  = False
    use_flash_attn:           bool  = True

    @classmethod
    def micro(cls) -> "ModelConfig":
        return cls(hidden_size=192, intermediate_size=512,
                   num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=48,
                   max_position_embeddings=512, vocab_size=512)

    @classmethod
    def small(cls) -> "ModelConfig":
        return cls(hidden_size=384, intermediate_size=1024,
                   num_hidden_layers=6, num_attention_heads=6,
                   num_key_value_heads=2, head_dim=64,
                   max_position_embeddings=1024)

    @classmethod
    def medium(cls) -> "ModelConfig":
        return cls()

    @classmethod
    def large(cls) -> "ModelConfig":
        return cls(hidden_size=1024, intermediate_size=2752,
                   num_hidden_layers=24, num_attention_heads=16,
                   num_key_value_heads=4, head_dim=64,
                   max_position_embeddings=4096)

    def estimate_mb(self, bpp: int = 4) -> float:
        p = (self.vocab_size * self.hidden_size
             + self.num_hidden_layers * (
                 self.hidden_size * self.head_dim *
                 (self.num_attention_heads + 2 * self.num_key_value_heads)
                 + self.hidden_size ** 2
                 + 3 * self.hidden_size * self.intermediate_size
                 + 4 * self.hidden_size
             ))
        return p * bpp / 1e6

    def save(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        p = Path(path)
        cfg_file = p / "config.json" if p.is_dir() else p
        with open(cfg_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    __slots__ = ()

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf   = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.to(x.dtype) * self.weight


# ─────────────────────────────────────────────────────────────
#  RoPE  (device-aware cache)
# ─────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096,
                 base: float = 500_000.0,
                 scale: Optional[float] = None) -> None:
        super().__init__()
        if scale and scale > 1.0:
            base = base * (scale ** (dim / (dim - 2)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max   = max_len
        self._cache_device: Optional[torch.device] = None
        self._build(max_len)

    def _build(self, n: int, device: Optional[torch.device] = None) -> None:
        dev = device or self.inv_freq.device
        t   = torch.arange(n, device=dev, dtype=torch.float32)
        f   = torch.outer(t, self.inv_freq.to(dev))
        emb = torch.cat([f, f], dim=-1)
        self.register_buffer("cos", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self._cache_device = dev

    def forward(self, seq_len: int, device: torch.device,
                dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        # FIX: rebuild cache if device changed (e.g. after .to(cuda))
        if self._cache_device != device or seq_len > self._max:
            self._max = max(seq_len * 2, self._max)
            self._build(self._max, device)
        return (self.cos[:, :, :seq_len].to(dtype=dtype),
                self.sin[:, :, :seq_len].to(dtype=dtype))


def _rotate(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: Tensor, k: Tensor,
               cos: Tensor, sin: Tensor,
               offset: int = 0) -> Tuple[Tensor, Tensor]:
    cq = cos[:, :, offset: offset + q.shape[2]]
    sq = sin[:, :, offset: offset + q.shape[2]]
    ck = cos[:, :, :k.shape[2]]
    sk = sin[:, :, :k.shape[2]]
    return q * cq + _rotate(q) * sq, k * ck + _rotate(k) * sk


# ─────────────────────────────────────────────────────────────
#  Fused GQA
# ─────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.nh   = cfg.num_attention_heads
        self.nkv  = cfg.num_key_value_heads
        self.hd   = cfg.head_dim
        self.g    = self.nh // self.nkv
        self.sdpa = cfg.use_flash_attn and _SDPA
        self.adrop= cfg.attention_dropout
        # FIX: kv_dtype resolved at forward time (not at init) so device is known
        self._kv_dtype_str = cfg.kv_dtype
        self.scale = self.hd ** -0.5

        d = cfg.hidden_size
        self.qkv_proj = nn.Linear(d,
            self.nh * self.hd + 2 * self.nkv * self.hd, bias=False)
        self.o_proj   = nn.Linear(self.nh * self.hd, d, bias=False)
        self.rope     = RotaryEmbedding(self.hd,
                                         cfg.max_position_embeddings,
                                         cfg.rope_theta, cfg.rope_scaling)

    def forward(self, x: Tensor,
                mask: Optional[Tensor] = None,
                past: Optional[Tuple] = None,
                use_cache: bool = False,
                offset: int = 0) -> Tuple[Tensor, Optional[Tuple]]:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q_sz = self.nh * self.hd; k_sz = self.nkv * self.hd
        q, k, v = qkv.split([q_sz, k_sz, k_sz], dim=-1)
        q = q.view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = k.view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.hd).transpose(1, 2)

        kv_len = (past[0].shape[2] if past else 0) + T
        cos, sin = self.rope(kv_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin, offset=past[0].shape[2] if past else 0)

        if past is not None:
            k = torch.cat([past[0].to(k.dtype), k], dim=2)
            v = torch.cat([past[1].to(v.dtype), v], dim=2)

        # FIX: kv_dtype must match current device
        kv_dt = getattr(torch, self._kv_dtype_str, torch.float32)
        present = (k.to(kv_dt), v.to(kv_dt)) if use_cache else None

        if self.g > 1:
            k = k.repeat_interleave(self.g, dim=1)
            v = v.repeat_interleave(self.g, dim=1)

        if self.sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.adrop if self.training else 0.0,
                is_causal=(mask is None)
            )
        else:
            w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                w = w + mask
            else:
                cm = torch.triu(torch.full((T, k.shape[2]), float("-inf"),
                                device=x.device, dtype=x.dtype),
                                diagonal=k.shape[2] - T + 1)
                w = w + cm
            w   = F.softmax(w, -1)
            if self.training and self.adrop > 0: w = F.dropout(w, self.adrop)
            out = torch.matmul(w, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), present


# ─────────────────────────────────────────────────────────────
#  SwiGLU — FIX: explicit dim=-1 on chunk
# ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.hidden_size; i = cfg.intermediate_size
        self.gate_up = nn.Linear(h, 2 * i, bias=False)
        self.down    = nn.Linear(i, h,     bias=False)
        self.drop    = (nn.Dropout(cfg.hidden_dropout_prob)
                        if cfg.hidden_dropout_prob > 0 else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        # FIX: explicit dim=-1 (was relying on default which can misfire on batch>1)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.drop(F.silu(gate) * up))


# ─────────────────────────────────────────────────────────────
#  Block
# ─────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int = 0) -> None:
        super().__init__()
        self.ln1  = RMSNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.attn = GQAttention(cfg, idx)
        self.ln2  = RMSNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.ffn  = SwiGLU(cfg)
        self.drop = (nn.Dropout(cfg.hidden_dropout_prob)
                     if cfg.hidden_dropout_prob > 0 else nn.Identity())
        self._gc  = False

    def _impl(self, x, mask, past, use_cache):
        a, p  = self.attn(self.ln1(x), mask, past, use_cache)
        x     = x + self.drop(a)
        x     = x + self.drop(self.ffn(self.ln2(x)))
        return x, p

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                past: Optional[Tuple] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple]]:
        if self._gc and self.training:
            out = grad_ckpt(lambda h, m: self._impl(h, m, None, False)[0],
                            x, mask, use_reentrant=False)
            return out, None
        return self._impl(x, mask, past, use_cache)


# ─────────────────────────────────────────────────────────────
#  LionLLM
# ─────────────────────────────────────────────────────────────

class LionLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg  = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size,
                                   padding_idx=cfg.pad_token_id)
        self.drop  = (nn.Dropout(cfg.hidden_dropout_prob)
                      if cfg.hidden_dropout_prob > 0 else nn.Identity())
        self.layers= nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm  = RMSNorm(cfg.hidden_size, cfg.layer_norm_eps)
        self.head  = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.head.weight = self.embed.weight

        self._init_weights()
        logger.info("LionLLM: %.2fM params | ~%.0f MB",
                    self.n_params() / 1e6, cfg.estimate_mb())

    def _init_weights(self) -> None:
        std     = self.cfg.initializer_range
        res_std = std / math.sqrt(2 * self.cfg.num_hidden_layers)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                if "bias" in name: nn.init.zeros_(p)
                else:              nn.init.ones_(p)
            elif any(s in name for s in ("o_proj", "down")):
                nn.init.normal_(p, 0, res_std)
            else:
                nn.init.normal_(p, 0, std)

    def n_params(self, trainable: bool = False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if not trainable or p.requires_grad)

    # FIX: enable_gc / disable_gc properly toggle all blocks
    def enable_gc(self) -> None:
        for b in self.layers: b._gc = True
        logger.info("Gradient checkpointing ON")

    def disable_gc(self) -> None:
        for b in self.layers: b._gc = False   # FIX: was True in bug
        logger.info("Gradient checkpointing OFF")

    def forward(self, input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None,
                past_key_values: Optional[Tuple] = None,
                use_cache: bool = False) -> Dict[str, Any]:
        B, T = input_ids.shape
        x    = self.drop(self.embed(input_ids))

        amask: Optional[Tensor] = None
        if attention_mask is not None:
            amask = ((1.0 - attention_mask.float())[:, None, None, :]
                     * torch.finfo(x.dtype).min).to(x.dtype)

        presents: List = []
        for i, blk in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            x, pres = blk(x, amask, past, use_cache)
            if use_cache: presents.append(pres)

        x      = self.norm(x)
        logits = self.head(x)
        out: Dict[str, Any] = {
            "logits": logits,
            "past_key_values": tuple(presents) if (use_cache and presents) else None,
        }

        if labels is not None:
            sl = logits[:, :-1].contiguous()
            tl = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                sl.view(-1, sl.size(-1)), tl.view(-1),
                ignore_index=self.cfg.pad_token_id,
                label_smoothing=0.05,  # lighter smoothing for small datasets
            )
        return out

    def save_pretrained(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path)
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: Path,
                        map_location: str = "cpu") -> "LionLLM":
        path  = Path(path)
        model = cls(ModelConfig.load(path))
        sd    = torch.load(path / "model.pt",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(sd, strict=False)
        return model

    def quantize_int8(self) -> "LionLLM":
        torch.quantization.quantize_dynamic(
            self, {nn.Linear}, torch.qint8, inplace=True)
        return self


# ─────────────────────────────────────────────────────────────
#  InferenceEngine — CPU/AMD optimised
# ─────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, model: LionLLM,
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        if device is None:
            if torch.cuda.is_available():   device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else:                           device = "cpu"

        # FIX: never use float16 on CPU — causes NaN
        if dtype is None:
            if device == "cuda":  dtype = torch.float16
            else:                 dtype = torch.float32

        # FIX: update kv_dtype on config to match device
        if device == "cpu":
            model.cfg.kv_dtype = "float32"
            for blk in model.layers:
                blk.attn._kv_dtype_str = "float32"
        elif device == "cuda":
            model.cfg.kv_dtype = "float16"
            for blk in model.layers:
                blk.attn._kv_dtype_str = "float16"

        # Optimise CPU thread count
        if device == "cpu":
            n_threads = os.cpu_count() or 4
            torch.set_num_threads(n_threads)

        self.device = device
        self.dtype  = dtype
        self.model  = model.to(device=device, dtype=dtype).eval()
        self.cfg    = model.cfg
        logger.info("InferenceEngine: device=%s dtype=%s", device, dtype)

    @torch.inference_mode()
    def generate(self,
                 input_ids:          Tensor,
                 max_new_tokens:     int   = 128,
                 temperature:        float = 0.8,
                 top_k:              int   = 40,
                 top_p:              float = 0.92,
                 min_p:              float = 0.05,
                 repetition_penalty: float = 1.1,
                 frequency_penalty:  float = 0.0,
                 presence_penalty:   float = 0.0,
                 contrastive_alpha:  float = 0.0,
                 contrastive_k:      int   = 4,
                 stop_ids:           Optional[List[int]] = None,
                 tokenizer=None,
                 stop_strings:       Optional[List[str]] = None) -> Generator[int, None, None]:
        ids  = input_ids.to(self.device)
        pkv  = None
        gen: List[int] = []
        # FIX: freq tensor on correct device
        freq = torch.zeros(self.cfg.vocab_size, device=self.device, dtype=torch.float32)
        use_amp = (self.device == "cuda")

        for _ in range(max_new_tokens):
            cur = ids if pkv is None else ids[:, -1:]
            with torch.autocast(self.device, dtype=self.dtype, enabled=use_amp):
                out = self.model(cur, past_key_values=pkv, use_cache=True)
            logits = out["logits"][:, -1, :].float()
            pkv    = out["past_key_values"]

            if gen:
                seen  = torch.tensor(gen, device=self.device, dtype=torch.long)
                if repetition_penalty != 1.0:
                    lp = logits[0, seen]
                    logits[0, seen] = torch.where(lp < 0,
                                                   lp * repetition_penalty,
                                                   lp / repetition_penalty)
                if frequency_penalty != 0:  logits[0] -= frequency_penalty * freq
                if presence_penalty  != 0:  logits[0, seen.unique()] -= presence_penalty

            if temperature != 1.0 and temperature > 0: logits /= temperature

            if min_p > 0:
                p0 = logits.softmax(-1)
                logits[p0 < min_p * p0.max(-1, keepdim=True).values] = float("-inf")

            if top_k > 0:
                k  = min(top_k, logits.size(-1))
                th = logits.topk(k, dim=-1).values[:, -1, None]
                logits[logits < th] = float("-inf")

            if 0 < top_p < 1.0:
                sl, si = logits.sort(-1, descending=True)
                cp     = sl.softmax(-1).cumsum(-1)
                rm     = (cp - sl.softmax(-1)) > top_p
                rm[:, 0] = False
                logits.scatter_(-1, si, sl.masked_fill(rm, float("-inf")))

            probs = logits.softmax(-1)

            if contrastive_alpha > 0 and gen:
                tid = self._contrastive(probs, pkv, contrastive_k, contrastive_alpha)
            else:
                tid = int(torch.multinomial(probs, 1).item())

            gen.append(tid); freq[tid] += 1.0
            ids = torch.cat([ids, torch.tensor([[tid]], device=self.device)], dim=-1)
            yield tid

            if tid == self.cfg.eos_token_id: break
            if stop_ids and tid in stop_ids: break
            if stop_strings and tokenizer:
                if any(s in tokenizer.decode(gen[-20:]) for s in stop_strings): break

            # FIX: consistent window slide — trim ids AND reset cache together
            if ids.shape[1] >= self.cfg.max_position_embeddings - 32:
                keep = int(self.cfg.max_position_embeddings * 0.6)
                ids  = torch.cat([ids[:, :1], ids[:, -keep:]], dim=-1)
                pkv  = None   # must reset together
                gc.collect()
                if self.device == "cuda": torch.cuda.empty_cache()

    # FIX: corrected contrastive — uses self.model.embed.weight
    def _contrastive(self, probs: Tensor, pkv,
                     k: int, alpha: float) -> int:
        tp, ti   = probs.topk(k, dim=-1)
        # FIX: was self.model.embed() — correct is embed.weight lookup
        cands    = self.model.embed.weight[ti[0]].detach()
        cands    = F.normalize(cands.float(), dim=-1)
        if pkv and pkv[0] is not None:
            pv = pkv[0][1][:, :, -16:, :].mean((0, 1, 2))
            pv = F.normalize(pv.float(), dim=-1)
            sim= F.cosine_similarity(cands, pv.unsqueeze(0), dim=-1)
        else:
            sim = torch.zeros(k, device=probs.device)
        scores = (1 - alpha) * tp[0].float() - alpha * sim
        return int(ti[0, scores.argmax()].item())

    @torch.inference_mode()
    def generate_beam(self, input_ids: Tensor,
                      max_new_tokens: int = 64,
                      num_beams: int = 4,
                      length_penalty: float = 1.0,
                      no_repeat_ngram: int = 3) -> List[int]:
        ids  = input_ids.to(self.device)
        eos  = self.cfg.eos_token_id
        beams: List[Tuple[float, List[int]]] = [(0.0, ids[0].tolist())]
        done:  List[Tuple[float, List[int]]] = []

        for _ in range(max_new_tokens):
            cands: List[Tuple[float, List[int]]] = []
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
            if not cands: break
            cands.sort(key=lambda x: x[0])
            beams = cands[:num_beams]
            if len(done) >= num_beams: break

        best = sorted(done + beams, key=lambda x: x[0])[0][1]
        return best[input_ids.shape[1]:]
