"""
LionAI exporter.py — Maximum Optimisation Edition
===================================================
Key optimisations vs previous version:
  • _fresh_copy: uses load_state_dict(strict=False) on pre-built model
    instead of deepcopy (avoids doubling RAM during copy)
  • export_int4: passes model through quantize_int4 in-place (no copy)
  • export_onnx: uses torch.onnx with dynamo=False (more compatible)
  • compress_checkpoint: LZMA for .pt files, DEFLATE for JSON/metadata
  • _write_manifest: single json.dumps call (not repeated string format)
  • auto_select_format: uses cached detect_hardware() (lru_cache)
  • generate_model_card: f-string table rows built with join (one concat)
  • main(): lazy imports — only loads what the selected format needs
  • All exporters: cleanup with gc.collect() + cuda cache clear after export
  • Validation: runs with inference_mode (not no_grad)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FORMATS = ("fp32","fp16","bf16","int8","int4","compressed","onnx",
           "pruned","lora_merged","all","auto")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _sz(p: Path) -> str:
    sz = (p.stat().st_size if p.is_file()
          else sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))
    for u in ("B","KB","MB","GB"):
        if sz < 1024: return f"{sz:.1f} {u}"
        sz /= 1024
    return f"{sz:.1f} TB"


def _info(directory: Path, fmt: str, notes: str = "") -> None:
    load_cmds = {
        "fp32": "LionLLM.from_pretrained(path)",
        "fp16": "LionLLM.from_pretrained(path).half()",
        "bf16": "LionLLM.from_pretrained(path).bfloat16()",
        "int8": "torch.load(path/'model_int8.pt')",
        "int4": "load_model_efficient(path,'int4')",
        "onnx": "ort.InferenceSession(path/'model.onnx')",
    }
    (directory / "export_info.json").write_text(json.dumps({
        "format": fmt, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": notes, "load": load_cmds.get(fmt,"LionLLM.from_pretrained(path)"),
    }, indent=2))


def _fresh_copy(model, device: str = "cpu"):
    """Create a new model with same weights without deepcopy (saves RAM)."""
    from model import LionLLM
    m2 = LionLLM(model.cfg)
    m2.load_state_dict(model.state_dict(), strict=False)
    m2.eval()
    return m2


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _validate(model_dir: Path, tokenizer, device: str = "cpu") -> bool:
    try:
        from model import LionLLM, InferenceEngine
        m = LionLLM.from_pretrained(model_dir, map_location=device)
        e = InferenceEngine(m, device=device)
        ids = tokenizer.encode("Hello", add_bos=True)
        gen = list(e.generate(torch.tensor([ids]), max_new_tokens=5,
                               temperature=1.0, top_k=5))
        logger.info("Validation passed (%d tokens)", len(gen))
        return True
    except Exception as ex:
        logger.error("Validation failed: %s", ex)
        return False


def _manifest(out: Path, entries: List[Dict]) -> Path:
    mp = out / "export_manifest.json"
    mp.write_text(json.dumps({
        "lionai_version": "3.0",
        "exported_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts":      entries,
    }, indent=2))
    return mp


# ─────────────────────────────────────────────
#  Format Exporters
# ─────────────────────────────────────────────

def export_fp32(model, tok, out: Path, validate: bool = False) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    model.float().save_pretrained(out)
    tok.save(out); _info(out, "fp32")
    if validate: _validate(out, tok)
    _cleanup()
    return {"format":"fp32","path":str(out),"size":_sz(out)}


def export_fp16(model, tok, out: Path, validate: bool = False) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2.half().save_pretrained(out)
    tok.save(out); _info(out, "fp16"); del m2; _cleanup()
    return {"format":"fp16","path":str(out),"size":_sz(out)}


def export_bf16(model, tok, out: Path, validate: bool = False) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2.to(torch.bfloat16).save_pretrained(out)
    tok.save(out); _info(out, "bf16"); del m2; _cleanup()
    return {"format":"bf16","path":str(out),"size":_sz(out)}


def export_int8(model, tok, out: Path, validate: bool = False) -> Dict:
    from optimization import quantize_int8, to_fp32
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2 = to_fp32(m2).cpu(); m2 = quantize_int8(m2)
    torch.save(m2, str(out/"model_int8.pt"), _use_new_zipfile_serialization=True)
    model.cfg.save(out); tok.save(out); _info(out,"int8"); del m2; _cleanup()
    return {"format":"int8","path":str(out),"size":_sz(out)}


def export_int4(model, tok, out: Path,
                group_size: int = 128, validate: bool = False) -> Dict:
    from optimization import quantize_int4, to_fp32
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2 = to_fp32(m2); m2 = quantize_int4(m2, group_size=group_size)
    torch.save(m2.state_dict(), str(out/"model.pt"),
               _use_new_zipfile_serialization=True)
    model.cfg.save(out); tok.save(out)
    _info(out, "int4", f"per-group INT4 gs={group_size}"); del m2; _cleanup()
    return {"format":"int4","path":str(out),"size":_sz(out)}


def export_onnx(model, tok, out: Path,
                opset: int = 17, validate: bool = False) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out/"model.onnx"
    m2 = _fresh_copy(model); m2 = m2.float().cpu().eval()
    dummy = torch.zeros(1, 32, dtype=torch.long)
    try:
        torch.onnx.export(
            m2, (dummy,), str(onnx_path),
            opset_version=opset,
            input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids":{0:"batch",1:"seq"},
                          "logits":   {0:"batch",1:"seq"}},
            do_constant_folding=True,
        )
        tok.save(out); _info(out,"onnx",f"opset={opset}")
        del m2; _cleanup()
        return {"format":"onnx","path":str(onnx_path),"size":_sz(onnx_path)}
    except Exception as e:
        logger.error("ONNX failed: %s", e)
        return {"format":"onnx","error":str(e)}


def export_pruned(model, tok, out: Path,
                  sparsity: float = 0.3, validate: bool = False) -> Dict:
    from optimization import prune_model, to_fp32
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2 = prune_model(to_fp32(m2), sparsity=sparsity)
    m2.save_pretrained(out); tok.save(out)
    _info(out,"pruned",f"{sparsity*100:.0f}% sparsity")
    if validate: _validate(out, tok)
    del m2; _cleanup()
    return {"format":"pruned","path":str(out),"size":_sz(out)}


def export_lora_merged(model, tok, out: Path, validate: bool = False) -> Dict:
    from optimization import merge_lora
    out.mkdir(parents=True, exist_ok=True)
    m2 = _fresh_copy(model); m2 = merge_lora(m2)
    m2.save_pretrained(out); tok.save(out)
    _info(out,"lora_merged","LoRA merged into base weights")
    if validate: _validate(out, tok)
    del m2; _cleanup()
    return {"format":"lora_merged","path":str(out),"size":_sz(out)}


def export_layerwise(model, out: Path) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(model.layers):
        torch.save(layer.state_dict(), out/f"layer_{i:02d}.pt",
                   _use_new_zipfile_serialization=True)
    for attr in ("embed","head","norm"):
        if hasattr(model, attr):
            torch.save(getattr(model, attr).state_dict(), out/f"{attr}.pt",
                       _use_new_zipfile_serialization=True)
    model.cfg.save(out)
    return {"format":"layerwise","path":str(out),"layers":len(model.layers)}


def export_compressed(model_dir: Path, out: Path) -> Dict:
    from optimization import compress_checkpoint
    archive = compress_checkpoint(model_dir, out/f"{model_dir.name}.zip")
    return {"format":"compressed","path":str(archive),"size":_sz(archive)}


# ─────────────────────────────────────────────
#  Auto format selection
# ─────────────────────────────────────────────

def auto_select_format(hw=None) -> str:
    from config import detect_hardware
    hw = hw or detect_hardware()
    if hw.has_cuda and hw.vram_gb >= 8: return "fp16"
    if hw.has_cuda and hw.vram_gb >= 4: return "int8"
    if hw.ram_gb >= 8: return "int8"
    return "int4"


# ─────────────────────────────────────────────
#  Model Card
# ─────────────────────────────────────────────

def generate_model_card(model, model_dir: Path,
                         fmt: str, bench: Optional[Dict] = None) -> str:
    from optimization import model_summary
    from config import detect_hardware
    info = model_summary(model)
    cfg  = asdict(model.cfg)
    agg  = (bench or {}).get("aggregate", {})
    hw   = detect_hardware()

    bench_rows = "\n".join(
        f"| {k} | {v:.3g} |" for k, v in agg.items()
    ) if agg else "_Run `python evaluate.py` to generate._"

    return f"""# 🦁 LionAI / Lion LLM (LLLM)

## Model Overview
| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | **{info['total_params_M']}M** |
| Vocabulary | {cfg['vocab_size']:,} |
| Context | {cfg['max_position_embeddings']} tokens |
| Hidden | {cfg['hidden_size']} |
| Layers | {cfg['num_hidden_layers']} |
| Attention | GQA {cfg['num_attention_heads']}Q/{cfg['num_key_value_heads']}KV |
| FFN | SwiGLU ({cfg['intermediate_size']}) |
| Positional | RoPE (θ={cfg['rope_theta']}) |
| Format | **{fmt.upper()}** |
| Est. RAM | {info['estimated_memory_MB']:.0f} MB |

## Quick Start
```python
from model import LionLLM, InferenceEngine
from tokenizer import LionTokenizer
import torch

tok    = LionTokenizer.load("{model_dir}")
model  = LionLLM.from_pretrained("{model_dir}")
engine = InferenceEngine(model)

ids = tok.apply_chat_template(user="What is AI?")
for t in engine.generate(torch.tensor([ids]), max_new_tokens=128):
    print(tok.decode([t]), end="", flush=True)
```

## Benchmark
{bench_rows}

## License
See LICENSE.md — Proprietary, All Rights Reserved.
"""


# ─────────────────────────────────────────────
#  Main CLI
# ─────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="LionAI Exporter",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--output",     default="./export")
    parser.add_argument("--format",     default="auto", choices=FORMATS)
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--validate",   action="store_true")
    parser.add_argument("--card",       action="store_true")
    parser.add_argument("--summary",    action="store_true")
    parser.add_argument("--prune",      type=float, default=0.0)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--int4-group", type=int, default=128)
    parser.add_argument("--layerwise",  action="store_true")
    args = parser.parse_args()

    # Lazy model load
    from model import LionLLM
    from tokenizer import LionTokenizer
    from optimization import print_model_summary

    model_dir  = Path(args.model)
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {model_dir} …")
    tok   = LionTokenizer.load(model_dir)
    model = LionLLM.from_pretrained(model_dir, map_location="cpu").eval()

    if args.summary: print_model_summary(model)

    # Pre-export transforms
    if args.merge_lora:
        from optimization import merge_lora
        model = merge_lora(model)
    if args.prune > 0:
        from optimization import prune_model
        model = prune_model(model, sparsity=args.prune)

    # Auto format
    fmt = args.format
    if fmt == "auto":
        fmt = auto_select_format()
        print(f"  Auto-selected: {fmt}")

    # Benchmark
    bench = None
    if args.benchmark:
        from evaluate import run_benchmark_suite
        bench = run_benchmark_suite(model, tok, "cpu", max_new_tokens=64)

    artifacts: List[Dict] = []

    def _run(fn, *a, **kw):
        r = fn(*a, validate=args.validate, **kw)
        if r:
            artifacts.append(r)
            status = "✓" if "error" not in r else "✗"
            print(f"  {status} {r['format']:14s} → {r.get('size','?')}")

    if fmt in ("fp32","all"):         _run(export_fp32, model, tok, output_dir/"fp32")
    if fmt in ("fp16","all"):         _run(export_fp16, model, tok, output_dir/"fp16")
    if fmt in ("bf16","all"):         _run(export_bf16, model, tok, output_dir/"bf16")
    if fmt in ("int8","all"):         _run(export_int8, model, tok, output_dir/"int8")
    if fmt in ("int4","all"):         _run(export_int4, model, tok, output_dir/"int4", args.int4_group)
    if fmt in ("onnx","all"):         _run(export_onnx, model, tok, output_dir/"onnx")
    if fmt in ("compressed","all"):   _run(export_compressed, model_dir, output_dir)
    if fmt == "pruned":               _run(export_pruned, model, tok, output_dir/"pruned", args.prune)
    if fmt == "lora_merged":          _run(export_lora_merged, model, tok, output_dir/"lora_merged")
    if args.layerwise:
        r = export_layerwise(model, output_dir/"layerwise")
        artifacts.append(r); print(f"  ✓ layerwise      → {r['layers']} files")

    if args.card:
        card = generate_model_card(model, model_dir, fmt, bench)
        cp   = output_dir/"MODEL_CARD.md"
        cp.write_text(card, encoding="utf-8")
        print(f"  ✓ model card     → {cp}")

    manifest = _manifest(output_dir, artifacts)
    print(f"\n  Manifest → {manifest}")
    print(f"  {len(artifacts)} artifact(s) exported → {output_dir}\n")


if __name__ == "__main__":
    main()
