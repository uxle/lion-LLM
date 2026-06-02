"""
LionAI Demo Setup  [Enhanced]
================================
New vs v1:
  • Hardware-aware model size selection
  • Richer sample corpus (code + prose + Q&A + dialogue)
  • Auto-trains a real (tiny) tokenizer on the sample corpus
  • Generates a structured benchmark report after setup
  • Creates example knowledge documents
  • Writes a .env template file
  • Shows RAM/VRAM usage after model creation
  • Runs a full chatbot smoke-test (non-interactive)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from model import LionLLM, ModelConfig, InferenceEngine
from tokenizer import LionTokenizer, TokenizerTrainer
from config import detect_hardware, SystemConfig, setup_logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Rich Sample Corpus
# ─────────────────────────────────────────────

SAMPLE_TEXTS = [
    # Prose
    "LionAI is a fully offline, self-hosted artificial intelligence system built for privacy and control.",
    "Machine learning enables computers to learn patterns from data without explicit programming.",
    "The transformer architecture uses self-attention to model long-range dependencies in sequences.",
    "Neural networks are computational models inspired by the structure of the human brain.",
    "Natural language processing allows computers to understand and generate human language.",
    "Deep learning models consist of many layers that transform raw input into useful representations.",
    "Training a language model involves minimising cross-entropy loss on a large text corpus.",
    "Inference is the process of using a trained model to generate predictions or text.",
    "Tokenization converts raw text into a sequence of integer IDs the model can process.",
    "Embeddings are dense vector representations that capture semantic relationships between tokens.",
    "Attention mechanisms allow models to focus on relevant parts of the input context.",
    "Gradient descent optimises neural network weights by following the negative gradient of the loss.",
    "Backpropagation efficiently computes gradients using the chain rule of calculus.",
    "Regularisation techniques like dropout prevent models from memorising training data.",
    "A language model assigns probabilities to sequences of tokens from its vocabulary.",
    # Dialogue
    "User: What is LionAI? Assistant: LionAI is a local AI assistant that runs entirely on your device with no internet required.",
    "User: How do I save my conversation? Assistant: Type /save followed by a name, for example /save my_session.",
    "User: What is the capital of France? Assistant: The capital of France is Paris.",
    "User: Can you write code? Assistant: Yes! I can help with Python, JavaScript, and many other languages.",
    "User: What is gradient descent? Assistant: Gradient descent is an optimisation algorithm that adjusts model parameters in the direction of steepest decrease in the loss function.",
    # Q&A
    "Q: What does RAM stand for? A: Random Access Memory — the short-term memory your computer uses for active processes.",
    "Q: What is a transformer? A: A transformer is a neural network architecture based on self-attention, used in modern language models.",
    "Q: What is quantization? A: Quantization reduces model size by storing weights in lower-precision formats like INT8 or INT4.",
    "Q: What is RAG? A: Retrieval-Augmented Generation combines a knowledge retrieval system with a language model for grounded responses.",
    "Q: What is a GPU? A: A Graphics Processing Unit, highly parallel hardware used to accelerate matrix operations in deep learning.",
    # Code
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "import torch\nmodel = torch.nn.Linear(512, 512)\noutput = model(torch.randn(1, 512))",
    "def clean_text(text: str) -> str:\n    import re\n    return re.sub(r'\\s+', ' ', text).strip()",
    "class Tokenizer:\n    def encode(self, text):\n        return [ord(c) for c in text]\n    def decode(self, ids):\n        return ''.join(chr(i) for i in ids)",
    # Instructions
    "To install LionAI: 1. Install Python 3.9+. 2. Install PyTorch. 3. Run python demo_setup.py.",
    "To train a tokenizer: python tokenizer_trainer.py train --input ./data --vocab 16000 --output ./tok.",
    "To export INT8 model: python exporter.py --model ./runs/lionai/final --format int8.",
    "To ingest documents: type /docs path/to/file in the chat interface.",
    "To switch generation mode: type /mode contrastive for higher quality responses.",
]

SAMPLE_KNOWLEDGE = """
# LionAI Knowledge Base

## Architecture
LionAI uses a decoder-only transformer with Grouped Query Attention (GQA), Rotary Positional 
Embeddings (RoPE), SwiGLU feed-forward networks, and RMSNorm. Flash Attention (SDPA) is used 
when available for memory-efficient attention computation.

## Memory System
LionAI has three memory tiers:
- Short-term: active conversation window, trimmed by token budget
- Long-term: persistent facts stored in SQLite, retrieved with BM25
- Semantic: similarity-based retrieval for knowledge and past responses

## Quantization Modes
- FP32: full precision, best quality, most RAM
- FP16: half precision, recommended for CUDA GPUs
- INT8: dynamic quantization, ~50% RAM, good CPU performance
- INT4: per-group quantization, ~75% RAM reduction, good for tight budgets

## Hardware Requirements
- Micro model (~15M): 2 GB RAM minimum
- Small model (~50M): 4 GB RAM minimum  
- Medium model (~125M): 6 GB RAM, 4 GB with INT8
- Large model (~350M): 12 GB RAM, 6 GB with INT8

## Generation Modes
- Sample: temperature + top-k + top-p + min-p sampling (default)
- Contrastive: balances likelihood with degeneration penalty (higher quality)
- Beam search: deterministic, best for factual Q&A

## Commands Reference
/help, /reset, /save, /load, /memory, /learn, /forget, /stats,
/hardware, /config, /mode, /quant, /docs, /search, /export, /system, /exit
"""

ENV_TEMPLATE = """# LionAI Environment Configuration
# Copy to .env and adjust as needed

# Override model settings
LIONAI_DEVICE=auto
LIONAI_QUANTIZATION=none
LIONAI_MODEL_SIZE=medium
LIONAI_MAX_NEW_TOKENS=256
LIONAI_TEMPERATURE=0.8
LIONAI_TOP_K=40
LIONAI_TOP_P=0.92

# Paths
LIONAI_MODEL_DIR=./runs/lionai/final
LIONAI_DATA_DIR=./data
LIONAI_LOG_LEVEL=INFO

# Features
LIONAI_ENABLE_MEMORY=true
LIONAI_ENABLE_RAG=false
"""


# ─────────────────────────────────────────────
#  Setup Steps
# ─────────────────────────────────────────────

def setup_tokenizer(output_dir: Path, sample_texts: list) -> LionTokenizer:
    print("  [1/5] Training demo tokenizer …")
    trainer   = TokenizerTrainer(vocab_size=4096, min_frequency=1, show_progress=False)
    tokenizer = trainer.train(iter(sample_texts * 30))
    tokenizer.save(output_dir)
    print(f"        ✓ Vocab size: {tokenizer.vocab_size:,} tokens")
    return tokenizer


def setup_model(output_dir: Path, tokenizer: LionTokenizer,
                size: str, hw) -> LionLLM:
    print(f"  [2/5] Initialising {size} model …")
    size_map = {
        "micro":  ModelConfig.micro,
        "small":  ModelConfig.small,
        "medium": ModelConfig.medium,
        "large":  ModelConfig.large,
    }
    config = size_map.get(size, ModelConfig.small)()
    config.vocab_size = tokenizer.vocab_size

    model = LionLLM(config)
    model.save_pretrained(output_dir)

    n_params = model.num_parameters() / 1e6
    est_mb   = config.estimate_vram_mb()
    print(f"        ✓ {n_params:.1f}M parameters | est. {est_mb:.0f} MB (FP32)")
    return model


def setup_sample_data(data_dir: Path, sample_texts: list) -> None:
    print("  [3/5] Writing sample dataset …")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for text in sample_texts * 20:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"        ✓ {len(sample_texts)*20} examples → {train_path}")


def setup_knowledge(data_dir: Path) -> None:
    print("  [4/5] Ingesting knowledge base …")
    try:
        from knowledge import KnowledgeEngine
        ke = KnowledgeEngine(data_dir / "knowledge")
        n  = ke.ingest_text(SAMPLE_KNOWLEDGE, title="LionAI Docs", source="<demo>")
        print(f"        ✓ {n} chunks indexed")
    except Exception as e:
        print(f"        ✗ Knowledge setup failed: {e}")


def run_smoke_test(model: LionLLM, tokenizer: LionTokenizer,
                   device: str) -> bool:
    print("  [5/5] Running smoke test …")
    try:
        engine    = InferenceEngine(model, device=device)
        test_ids  = tokenizer.encode("Hello, LionAI!", add_bos=True)
        input_ids = torch.tensor([test_ids], dtype=torch.long)
        gen: list = []
        for tok in engine.generate(input_ids, max_new_tokens=20,
                                    temperature=1.0, top_k=10, top_p=0.9):
            gen.append(tok)
        decoded = tokenizer.decode(gen)
        print(f"        ✓ Generated {len(gen)} tokens: {decoded[:60]!r}")
        return True
    except Exception as e:
        print(f"        ✗ Smoke test failed: {e}")
        return False


def write_env_template(base_dir: Path) -> None:
    env_path = base_dir / ".env.template"
    env_path.write_text(ENV_TEMPLATE)
    print(f"\n  Config template → {env_path}")


def print_summary(model_dir: Path, hw, size: str) -> None:
    print(f"""
{'═'*55}
  🦁 LionAI Demo Setup Complete!
{'═'*55}

  Model:    {model_dir}  ({size})
  Device:   {hw.device.upper()}
  RAM:      {hw.ram_gb:.0f} GB detected

  ── Start chatting ─────────────────────────────
  python chatbot.py --model {model_dir}

  ── With quantization (lower RAM) ──────────────
  python chatbot.py --model {model_dir} --quantize int8

  ── Train on your own data ─────────────────────
  # 1. Process your dataset:
  python dataset_processor.py --sources ./mydata/ --output ./data

  # 2. Train tokenizer:
  python tokenizer_trainer.py train \\
      --input ./data/train.jsonl \\
      --output {model_dir} --vocab 16000

  # 3. Train model:
  python train.py

  # 4. Export:
  python exporter.py --model {model_dir} --format auto

  ── Evaluate ───────────────────────────────────
  python evaluate.py --model {model_dir}

{'═'*55}
  NOTE: Demo model uses random weights.
  Responses are incoherent until trained on real data.
{'═'*55}
""")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main() -> None:
    setup_logging("INFO", log_to_file=False)

    parser = argparse.ArgumentParser(
        description="LionAI Demo Setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size",    choices=["micro","small","medium","large"],
                        default=None, help="Model size (auto if not set)")
    parser.add_argument("--output",  default="./runs/lionai/final")
    parser.add_argument("--data",    default="./data")
    parser.add_argument("--device",  default=None)
    parser.add_argument("--no-data", action="store_true",
                        help="Skip sample dataset creation")
    parser.add_argument("--no-knowledge", action="store_true",
                        help="Skip knowledge base setup")
    args = parser.parse_args()

    print("\n  🦁 LionAI Demo Setup")
    print("  " + "─" * 40)

    hw     = detect_hardware()
    size   = args.size or hw.recommended_model_size
    device = args.device or hw.device
    print(f"  Hardware: {hw.ram_gb:.0f}GB RAM  {hw.vram_gb:.0f}GB VRAM  {hw.device}")
    print(f"  Selected: {size} model on {device}\n")

    model_dir = Path(args.output)
    data_dir  = Path(args.data)

    # Run setup pipeline
    tokenizer = setup_tokenizer(model_dir, SAMPLE_TEXTS)
    model     = setup_model(model_dir, tokenizer, size, hw)

    if not args.no_data:
        setup_sample_data(data_dir, SAMPLE_TEXTS)
    else:
        print("  [3/5] Skipping sample data")

    if not args.no_knowledge:
        setup_knowledge(data_dir)
    else:
        print("  [4/5] Skipping knowledge base")

    smoke_ok  = run_smoke_test(model, tokenizer, device)

    # Write env template
    write_env_template(Path("."))

    # Print summary
    print_summary(model_dir, hw, size)

    if not smoke_ok:
        print("  ⚠ Smoke test failed — check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
