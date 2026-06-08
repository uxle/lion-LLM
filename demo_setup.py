"""
LionAI demo_setup.py — Bug-Fixed + AMD/CPU-Optimised Edition
=============================================================
Bugs fixed:
  BUG 1: vocab_size=4096 for 50 words → thousands of useless merges (9 min wait)
          → auto_vocab_size() picks vocab proportional to corpus size
  BUG 2: max_seq_length=512 with 50 words → 0 training examples
          → auto_seq_length() picks seq proportional to token count
  BUG 3: No feedback during tokenizer training (looked frozen)
          → progress printed every 10% with ETA
  BUG 4: Model always created with random weights; no check if already exists
          → skip model creation if checkpoint already present
  BUG 5: AMD RX550 has no CUDA → torch.cuda calls silently no-op or crash
          → all CUDA refs gated behind torch.cuda.is_available()
  BUG 6: demo_setup ran training automatically with bad defaults
          → setup only; training now done by train.py with proper auto-config
  BUG 7: No progress indicator for model initialisation (felt frozen on CPU)
          → print before + after with param count and estimated time
  BUG 8: knowledge base ingestion could fail silently
          → wrapped in try/except with clear user message
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from config import detect_hardware, setup_logging

logger = logging.getLogger(__name__)


# ── Sample corpus (rich enough to demonstrate the system) ────────────────────
_TEXTS = [
    "LionAI is a fully offline AI assistant running on your local device.",
    "Machine learning enables computers to find patterns in data automatically.",
    "The transformer architecture uses self-attention to model long sequences.",
    "Neural networks are computational models loosely inspired by the brain.",
    "Natural language processing helps computers read and write human text.",
    "Deep learning uses many layers to transform raw inputs into useful outputs.",
    "Training a model means adjusting its weights to reduce prediction error.",
    "Inference is when a trained model generates predictions on new input.",
    "Tokenization splits text into small units called tokens for processing.",
    "Embeddings map tokens to dense numerical vectors capturing meaning.",
    "Attention lets the model focus on relevant context when generating text.",
    "Gradient descent finds the direction that reduces the training loss.",
    "Backpropagation efficiently computes weight gradients using chain rule.",
    "Dropout randomly zeroes activations during training to reduce overfitting.",
    "A language model estimates the probability of the next token in a sequence.",
    "User: What is LionAI? Assistant: A local AI that runs entirely offline.",
    "User: How do I save a chat? Assistant: Type /save followed by a name.",
    "Q: What is RAM? A: Random Access Memory — fast short-term computer storage.",
    "Q: What is a GPU? A: A parallel processor great for matrix operations.",
    "Q: What is quantization? A: Storing weights in fewer bits to save memory.",
    "def hello(): return 'Hello, LionAI!'",
    "import torch; model = torch.nn.Linear(64, 64)",
    "To train: python train.py",
    "To chat: python chatbot.py --model ./runs/lionai/final",
    "LionAI supports CPU, CUDA, ROCm (AMD), and Apple MPS backends.",
]

_KNOWLEDGE_MD = """
# LionAI Quick Reference

## Starting the chatbot
```
python chatbot.py --model ./runs/lionai/final
```

## Training on your own data
```
python train.py --dataset ./data/train.jsonl
```

## Hardware support
- Intel/AMD CPU: always supported (uses all cores automatically)
- NVIDIA GPU: detected via CUDA
- AMD GPU: detected via ROCm (if PyTorch-ROCm installed)
- Apple Silicon: detected via MPS

## Generation modes
- /mode sample      — creative, varied responses (default)
- /mode contrastive — less repetition, higher quality
- /mode beam        — deterministic, best for factual Q&A

## Memory commands
- /learn KEY VALUE  — store a fact
- /memory           — list stored facts
- /forget KEY       — delete a fact

## Knowledge base
- /docs ./file.pdf  — ingest a document
- /search QUERY     — search indexed documents
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def auto_vocab_size(n_words: int) -> int:
    """
    Pick vocabulary size proportional to corpus size.
    Having vocab >> unique_words wastes training time with no benefit.
    """
    if n_words < 100:    return max(64,  n_words * 3)
    if n_words < 500:    return max(256, n_words * 2)
    if n_words < 2000:   return 2000
    if n_words < 10000:  return 8000
    return 32000


def auto_seq_length(n_tokens: int) -> int:
    """
    Pick sequence length so we actually get training examples.
    Rule: seq_len ≤ total_tokens / 4 (so we get ≥ 4 packed chunks).
    """
    if n_tokens < 50:   return max(8,  n_tokens // 2)
    if n_tokens < 200:  return max(16, n_tokens // 4)
    if n_tokens < 1000: return 64
    if n_tokens < 5000: return 128
    return 256


def count_words_and_tokens(texts: list) -> tuple:
    """Quick estimate of words and tokens in the corpus."""
    all_words: set = set()
    total_chars = 0
    for t in texts:
        all_words.update(t.lower().split())
        total_chars += len(t)
    # Rough: 4 chars ≈ 1 token
    return len(all_words), total_chars // 4


# ── Setup Steps ───────────────────────────────────────────────────────────────

def step_tokenizer(out: Path, texts: list,
                   vocab_size: int) -> "LionTokenizer":  # noqa: F821
    from tokenizer import LionTokenizer, TokenizerTrainer

    tok_path = out / "tokenizer.json"
    if tok_path.exists():
        print(f"        ↳ Already exists — loading")
        return LionTokenizer.load(out)

    print(f"        Vocabulary target: {vocab_size} tokens")
    print(f"        Corpus: {len(texts)} sentences")

    t0 = time.perf_counter()
    trainer   = TokenizerTrainer(
        vocab_size    = vocab_size,
        min_frequency = 1,
        show_progress = True,
    )
    tokenizer = trainer.train(iter(texts * 10))   # repeat to build good stats
    tokenizer.save(out)
    elapsed = time.perf_counter() - t0

    print(f"        ✓ {tokenizer.vocab_size:,} tokens  {len(tokenizer.merges):,} merges  ({elapsed:.1f}s)")
    return tokenizer


def step_model(out: Path, tokenizer, model_size: str) -> "LionLLM":  # noqa: F821
    from model import LionLLM, ModelConfig
    import dataclasses

    model_pt = out / "model.pt"
    if model_pt.exists():
        print(f"        ↳ Already exists — loading")
        return LionLLM.from_pretrained(out, map_location="cpu")

    cfg_fn = getattr(ModelConfig, model_size, ModelConfig.micro)
    cfg    = dataclasses.replace(cfg_fn(), vocab_size=tokenizer.vocab_size)

    est_mb = cfg.estimate_mb()
    print(f"        Size: {model_size}  ~{cfg.estimate_mb():.0f} MB FP32")

    t0 = time.perf_counter()
    with torch.no_grad():
        model = LionLLM(cfg)
    model.save_pretrained(out)
    elapsed = time.perf_counter() - t0

    print(f"        ✓ {model.n_params()/1e6:.1f}M params  ({elapsed:.1f}s)")
    return model


def step_dataset(data_dir: Path, texts: list, seq_len: int) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"

    if train_path.exists():
        print(f"        ↳ Already exists")
        return train_path

    # Write texts, repeat so we have enough tokens for packing
    repeats = max(20, 512 // max(len(texts[0]), 1))
    content = "\n".join(
        json.dumps({"text": t}) for t in texts * repeats
    )
    train_path.write_text(content + "\n", encoding="utf-8")
    n_tokens_est = sum(len(t) for t in texts) * repeats // 4
    print(f"        ✓ {len(texts)*repeats} lines  ~{n_tokens_est} tokens")
    return train_path


def step_knowledge(data_dir: Path) -> None:
    try:
        from knowledge import KnowledgeEngine
        ke = KnowledgeEngine(data_dir / "knowledge")
        if ke.stats().get("documents", 0) > 0:
            print("        ↳ Already indexed")
            return
        n = ke.ingest_text(_KNOWLEDGE_MD, title="LionAI Docs", source="<demo>")
        print(f"        ✓ {n} chunks indexed")
    except Exception as e:
        print(f"        ✗ Skipped ({e})")


def step_smoke_test(model, tokenizer, device: str) -> bool:
    from model import InferenceEngine
    try:
        engine    = InferenceEngine(model, device=device)
        input_ids = torch.tensor(
            [tokenizer.encode("Hello LionAI", add_bos=True)],
            dtype=torch.long
        )
        gen = list(engine.generate(input_ids, max_new_tokens=8,
                                    temperature=1.0, top_k=5))
        decoded = tokenizer.decode(gen)
        print(f"        ✓ {len(gen)} tokens generated (random weights — gibberish is expected)")
        return True
    except Exception as e:
        print(f"        ✗ {e}")
        return False


def print_next_steps(model_dir: Path, data_dir: Path,
                     hw, model_size: str) -> None:
    sep = "═" * 58
    quant_hint = ""
    if hw.vram_gb > 0 and hw.vram_gb < 6:
        quant_hint = f"\n  Low VRAM detected ({hw.vram_gb:.0f}GB) — use --quantize int8"
    elif hw.ram_gb < 8:
        quant_hint = f"\n  Low RAM detected ({hw.ram_gb:.0f}GB) — use --quantize int4"

    print(f"""
{sep}
  🦁  LionAI Demo Setup Complete!
{sep}

  Model:  {model_dir}  ({model_size})
  Device: {hw.device.upper()}  |  RAM: {hw.ram_gb:.0f}GB  |  VRAM: {hw.vram_gb:.0f}GB
{quant_hint}

  ── Step 1: Add your own training data ────────────────
  # Put .txt, .jsonl, .md, or .pdf files in ./mydata/
  python dataset_processor.py \\
      --sources ./mydata/ --output ./data

  ── Step 2: Train the model ───────────────────────────
  python train.py --dataset ./data/train.jsonl
  # (auto-detects vocab size, seq length, batch size)

  ── Step 3: Chat ──────────────────────────────────────
  python chatbot.py --model {model_dir}
  python chatbot.py --model {model_dir} --quantize int8

  ── Evaluate ──────────────────────────────────────────
  python evaluate.py --model {model_dir}

  ── Export ────────────────────────────────────────────
  python exporter.py --model {model_dir} --format auto

  NOTE: The demo model has random weights.
  Responses become coherent after training on real data.
{sep}
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging("INFO", log_to_file=False)

    parser = argparse.ArgumentParser(
        description="LionAI Demo Setup — creates a ready-to-train installation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output",       default="./runs/lionai/final",
                        help="Model output directory")
    parser.add_argument("--data",         default="./data",
                        help="Data directory")
    parser.add_argument("--model-size",   default=None,
                        choices=["micro", "small", "medium", "large"],
                        help="Model size (default: auto)")
    parser.add_argument("--vocab",        type=int, default=None,
                        help="Vocabulary size (default: auto)")
    parser.add_argument("--device",       default=None,
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--no-knowledge", action="store_true",
                        help="Skip knowledge base setup")
    parser.add_argument("--force",        action="store_true",
                        help="Re-create even if already exists")
    args = parser.parse_args()

    print("\n  🦁  LionAI Demo Setup")
    print("  " + "─" * 50)

    # Hardware detection
    hw     = detect_hardware()
    device = args.device or hw.device
    if device == "auto": device = hw.device

    print(f"\n  Hardware detected:")
    print(f"    CPU cores:  {hw.cpu_cores}")
    print(f"    RAM:        {hw.ram_gb:.1f} GB")
    print(f"    VRAM:       {hw.vram_gb:.1f} GB")
    print(f"    Device:     {hw.device.upper()}")
    if hw.has_cuda:
        try:
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"    GPU:        {gpu_name}")
        except Exception:
            pass
    print()

    # Auto-size based on demo corpus
    n_unique_words, n_tokens_est = count_words_and_tokens(_TEXTS)
    vocab_size = args.vocab or auto_vocab_size(n_unique_words)
    seq_len    = auto_seq_length(n_tokens_est)

    # Auto model size
    if args.model_size:
        model_size = args.model_size
    else:
        model_size = hw.recommended_model_size
        # For demo, always use micro (fastest setup)
        if model_size not in ("micro", "small"):
            model_size = "micro"

    print(f"  Auto-selected settings:")
    print(f"    Model size: {model_size}")
    print(f"    Vocab size: {vocab_size}")
    print(f"    Seq length: {seq_len}")
    print()

    model_dir = Path(args.output)
    data_dir  = Path(args.data)

    if args.force:
        import shutil
        if model_dir.exists(): shutil.rmtree(model_dir)
        print("  (--force: removed existing model)")

    # ── Step 1: Tokenizer ────────────────────────────────────────────────────
    print("  [1/5] Tokenizer")
    tokenizer = step_tokenizer(model_dir, _TEXTS, vocab_size)

    # ── Step 2: Model ────────────────────────────────────────────────────────
    print("  [2/5] Model")
    model = step_model(model_dir, tokenizer, model_size)

    # ── Step 3: Dataset ──────────────────────────────────────────────────────
    print("  [3/5] Dataset")
    step_dataset(data_dir, _TEXTS, seq_len)

    # ── Step 4: Knowledge ────────────────────────────────────────────────────
    print("  [4/5] Knowledge base")
    if not args.no_knowledge:
        step_knowledge(data_dir)
    else:
        print("        Skipped (--no-knowledge)")

    # ── Step 5: Smoke test ───────────────────────────────────────────────────
    print("  [5/5] Smoke test")
    ok = step_smoke_test(model, tokenizer, device)

    # Write env template
    env_template = (
        "# LionAI environment overrides\n"
        "# Rename to .env and adjust as needed\n"
        f"LIONAI_DEVICE={device}\n"
        f"LIONAI_QUANTIZATION={hw.recommended_quantization}\n"
        "LIONAI_MAX_NEW_TOKENS=256\n"
        "LIONAI_TEMPERATURE=0.8\n"
        "LIONAI_LOG_LEVEL=INFO\n"
    )
    Path(".env.template").write_text(env_template, encoding="utf-8")

    print_next_steps(model_dir, data_dir, hw, model_size)

    if not ok:
        print("  ⚠  Smoke test failed — check logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
