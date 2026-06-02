# LionAI — Lion LLM (LLLM)
### Next-Generation Independent Local AI System

```
  __  __ _       _ _ _     __  __
 |  \/  (_)_ __ (_) | |   |  \/  |
 | |\/| | | '_ \| | | |   | |\/| |
 | |  | | | | | | | | |___| |  | |
 |_|  |_|_|_| |_|_|_|_____|_|  |_|
```

**Powered by Lion LLM (LLLM) — a fully offline, self-hosted AI platform you build and own from scratch.**

LionAI is a complete, production-grade artificial intelligence system designed to run entirely on your local hardware. No cloud APIs. No subscriptions. No data leaving your machine. Every component — from the tokenizer to the memory system to the knowledge engine — is implemented from first principles and fully within your control.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Training Guide](#training-guide)
7. [Inference Guide](#inference-guide)
8. [Chat Interface](#chat-interface)
9. [Memory System](#memory-system)
10. [Knowledge System (RAG)](#knowledge-system-rag)
11. [Optimization Guide](#optimization-guide)
12. [Hardware Requirements](#hardware-requirements)
13. [Benchmarks](#benchmarks)
14. [Project Structure](#project-structure)
15. [Roadmap](#roadmap)
16. [License](#license)

---

## Overview

LionAI is built on the principle of **intelligence-per-parameter** — achieving maximum capability within a compact, resource-efficient design that runs on consumer hardware.

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameter range | 50M – 350M |
| Min RAM | 4 GB |
| Platform | Windows / macOS / Linux |
| Offline | Fully (zero internet required) |
| Dependencies | PyTorch only (core) |

### Design Principles

- **Fully offline** — Works with no network access
- **Self-hosted** — You own every component
- **Lightweight but scalable** — Runs on laptops; scales to servers
- **Resource-efficient** — Optimised for consumer hardware
- **Extensible** — Modular architecture for future additions
- **Secure by design** — Input validation, sandboxing, atomic writes
- **Educational** — Clean, documented, readable source code
- **Production-grade** — Error handling, logging, recovery throughout

---

## Features

### Core AI
- Custom BPE tokenizer trained on your data
- Modern decoder-only transformer (GQA + RoPE + SwiGLU + RMSNorm)
- Grouped-Query Attention for efficient inference
- Rotary Positional Embeddings for length generalisation
- SwiGLU feed-forward network (outperforms standard FFN)

### Training
- Full training loop with gradient accumulation
- Mixed-precision training (FP16/BF16)
- Cosine LR schedule with warm-up
- Early stopping and validation tracking
- Checkpoint save/resume system
- Custom dataset ingestion (txt, jsonl, md, csv, pdf)
- Dataset cleaning, deduplication, and validation

### Inference
- Streaming token generation
- Top-k, top-p (nucleus), and temperature sampling
- Repetition penalty
- KV-cache for fast autoregressive generation
- Context window management with sliding

### Memory
- Short-term: active conversation context
- Long-term: SQLite-backed persistent memory
- Semantic: similarity-based knowledge retrieval

### Knowledge (RAG)
- Offline document ingestion
- PDF, Markdown, TXT, JSONL support
- Full-text search via SQLite FTS5
- Chunk-based retrieval with overlap
- Automatic context augmentation

### Optimization
- FP16 / BF16 precision conversion
- INT8 dynamic quantization
- INT4 weight quantization (custom, no bitsandbytes)
- Compressed checkpoint export
- ONNX export for cross-platform deployment
- Weight tying

### Security
- Input validation and sanitization
- Path traversal prevention
- Atomic file writes (crash-safe)
- SHA-256 checksum verification
- Process lock file management
- Crash recovery system

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LIONAI SYSTEM                   │
├─────────────────┬───────────────┬───────────────────┤
│   TOKENIZER     │     MODEL     │     INFERENCE     │
│                 │               │                   │
│ BPE Vocabulary  │ GQA Attention │ KV-Cache          │
│ Byte Encoder    │ RoPE Embed.   │ Top-k/p Sampling  │
│ Subword Split   │ SwiGLU FFN    │ Streaming Output  │
│ Special Tokens  │ RMSNorm       │ Repetition Pen.   │
├─────────────────┴───────────────┴───────────────────┤
│                    MEMORY SYSTEM                    │
│  Short-Term     │  Long-Term    │  Semantic          │
│  (In-Memory)    │  (SQLite)     │  (TF-IDF Index)   │
├─────────────────────────────────────────────────────┤
│               KNOWLEDGE ENGINE (RAG)                │
│  Ingestion → Chunking → FTS5 Index → Retrieval     │
├─────────────────────────────────────────────────────┤
│                 OPTIMIZATION LAYER                  │
│       FP16 │ BF16 │ INT8 │ INT4 │ ONNX │ ZIP       │
└─────────────────────────────────────────────────────┘
```

### Transformer Block

```
Input
  │
  ├─[RMSNorm]─[Grouped-Query Attention]─[+]─ Residual
  │                (RoPE + GQA + KV-Cache)
  │
  └─[RMSNorm]─[SwiGLU FFN]─[+]──────────── Residual
                                    │
                                  Output
```

---

## Installation

### Prerequisites
- Python 3.9 or later
- PyTorch 2.1 or later

### Step 1: Clone / Download

```bash
# Place all files in a directory
mkdir lionai && cd lionai
# Copy all .py files here
```

### Step 2: Create Environment

```bash
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3: Install PyTorch

```bash
# CPU only:
pip install torch

# CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS):
pip install torch
```

### Step 4: Optional Dependencies

```bash
# For PDF support:
pip install pdfplumber

# For ONNX export:
pip install onnx onnxruntime
```

### Step 5: Verify Installation

```bash
python demo_setup.py --size small
```

---

## Quick Start

### Option A: Demo (Random Weights)

```bash
# Create a demo model (random weights — nonsensical output)
python demo_setup.py --size small

# Start chatting immediately
python chatbot.py --model ./runs/lionai/final
```

### Option B: Train Your Own Model

```bash
# 1. Prepare dataset
python dataset_processor.py \
    --sources ./my_text_data/ \
    --output ./data

# 2. Train tokenizer
python tokenizer_trainer.py train \
    --input ./data/train.jsonl \
    --output ./runs/lionai/final \
    --vocab_size 16000

# 3. Train model
python train.py

# 4. Chat
python chatbot.py --model ./runs/lionai/final
```

---

## Training Guide

### Dataset Preparation

LionAI accepts plain text, JSONL, Markdown, and CSV files.

**JSONL format (recommended):**
```json
{"text": "Your training text here."}
{"text": "Another document or paragraph."}
```

**Alpaca instruction format:**
```json
{"instruction": "Explain X", "input": "", "output": "X is ..."}
```

**Process your data:**
```bash
# From text files
python dataset_processor.py \
    --sources ./raw_docs/ ./books/ \
    --output ./data \
    --split 0.95

# From an Alpaca-style instruction dataset
python dataset_processor.py \
    --sources ./alpaca_data.json \
    --output ./data \
    --instruction
```

### Tokenizer Training

```bash
# Train on your corpus
python tokenizer_trainer.py train \
    --input ./data/train.jsonl \
    --output ./my_tokenizer \
    --vocab_size 32000 \
    --min_freq 2 \
    --analyze \
    --test
```

Vocabulary size guide:
| Corpus size | Recommended vocab |
|---|---|
| < 10MB | 4,000 – 8,000 |
| 10–100MB | 8,000 – 16,000 |
| 100MB–1GB | 16,000 – 32,000 |
| > 1GB | 32,000 – 64,000 |

### Model Configuration

Edit `train.py` or create a config JSON:

```json
{
  "model_size": "medium",
  "learning_rate": 3e-4,
  "batch_size": 8,
  "gradient_accumulation_steps": 4,
  "max_steps": 50000,
  "max_seq_length": 512,
  "warmup_steps": 200
}
```

Model sizes:

| Size | Parameters | RAM (FP32) | RAM (INT8) |
|---|---|---|---|
| small | ~50M | ~200 MB | ~100 MB |
| medium | ~125M | ~500 MB | ~250 MB |
| large | ~350M | ~1.4 GB | ~700 MB |

### Running Training

```bash
python train.py

# With custom config
python train.py \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --max_steps 100000

# Resume from checkpoint
python train.py --resume
```

### Monitoring

Training logs are written to `./runs/lionai/metrics.json`.

```python
import json
metrics = json.load(open("./runs/lionai/metrics.json"))
# Each entry: {"step": ..., "train_loss": ..., "val_loss": ..., "lr": ...}
```

---

## Inference Guide

### Python API

```python
from model import LionLLM, InferenceEngine
from tokenizer import LionTokenizer
import torch

# Load
tokenizer = LionTokenizer.load("./runs/lionai/final")
model     = LionLLM.from_pretrained("./runs/lionai/final")
engine    = InferenceEngine(model, device="cpu")

# Generate (streaming)
prompt    = "<sys>You are a helpful assistant.</sys>\n<usr>What is AI?</usr>\n<ast>"
input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)])

for token_id in engine.generate(
    input_ids,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
):
    print(tokenizer.decode([token_id]), end="", flush=True)
```

### Optimized Loading

```python
from optimization import load_model_efficient

# INT8 for CPU (2× faster, 2× smaller)
model = load_model_efficient("./runs/lionai/final", quantization="int8")

# INT4 for extreme compression
model = load_model_efficient("./runs/lionai/final", quantization="int4")
```

---

## Chat Interface

```bash
# Standard start
python chatbot.py --model ./runs/lionai/final

# With INT8 quantization
python chatbot.py --model ./runs/lionai/final --quantize

# Force CPU
python chatbot.py --model ./runs/lionai/final --device cpu

# Verbose mode
python chatbot.py --model ./runs/lionai/final --verbose
```

### Commands

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/reset` | Clear conversation history |
| `/save [name]` | Save current session |
| `/load [name]` | Load a saved session |
| `/memory` | List all stored memories |
| `/learn KEY VALUE` | Store a memory |
| `/forget KEY` | Delete a stored memory |
| `/stats` | Show model and session statistics |
| `/config KEY=VALUE` | Update inference settings |
| `/export [file]` | Export conversation to JSON |
| `/docs [path]` | Ingest document(s) into knowledge base |
| `/search QUERY` | Search the knowledge base |
| `/exit` | Exit and save session |

### Configuration at runtime

```
/config temp=0.7
/config top_k=40
/config top_p=0.85
/config max_tokens=512
```

---

## Memory System

LionAI uses a three-layer memory architecture:

### Short-Term Memory
Active conversation window. Automatically trimmed when the token budget is exceeded (oldest messages removed first, system prompt preserved).

```python
from memory import MemoryManager

mm = MemoryManager("./data/memory")
mm.short.add("user", "My name is Alice")
mm.short.add("assistant", "Hello Alice!")
print(mm.short.get_prompt())
```

### Long-Term Memory
Persistent facts and preferences stored in SQLite. Survives across sessions.

```python
mm.remember("user_name", "Alice", category="preference", importance=0.9)
mm.remember("user_location", "London", category="fact")

# Automatic retrieval by keyword
context = mm.recall("what is my name")
```

### Semantic Memory
Similarity-based retrieval using character n-gram TF-IDF (no external embedding model required).

```python
mm.semantic.add("Alice works as a software engineer in London")
results = mm.semantic.search("what does Alice do for work?")
```

### Memory in Chat
When you type `/learn name Alice` in the chat, the memory is:
1. Stored in long-term memory (SQLite)
2. Indexed in semantic memory (vector store)
3. Automatically retrieved on relevant queries

---

## Knowledge System (RAG)

Add your own documents to give LionAI domain-specific knowledge.

### Supported Formats
- Plain text (`.txt`)
- Markdown (`.md`)
- PDF (`.pdf`) — requires `pdfplumber`
- JSONL knowledge bases (`.jsonl`)

### Ingest Documents

```bash
# In the chat interface
/docs ./my_manual.pdf
/docs ./knowledge_base/

# Via Python
from knowledge import KnowledgeEngine
ke = KnowledgeEngine("./data/knowledge")
ke.ingest_file("./my_document.pdf")
ke.ingest_directory("./docs/")
```

### How RAG Works

```
User query: "How do I configure the system?"
     │
     ▼
Knowledge engine searches indexed chunks via FTS5
     │
     ▼
Top-k chunks retrieved and formatted as context
     │
     ▼
Context prepended to model prompt:
  [Knowledge Context]
  [Source: manual.pdf]
  Configuration is done via the config.json file…
  [End Context]
  <usr>How do I configure the system?</usr>
  <ast>
     │
     ▼
Model generates response grounded in the retrieved context
```

---

## Optimization Guide

### Choosing a Quantization Mode

| Mode | Memory | Speed | Quality | Best For |
|---|---|---|---|---|
| FP32 | 100% | baseline | Best | Training, highest accuracy |
| FP16 | 50% | 1.5–2× | Near-identical | CUDA GPU inference |
| BF16 | 50% | 1.5–2× | Near-identical | Modern GPUs, stability |
| INT8 | 50% | 1.5–2× | Very good | CPU deployment |
| INT4 | 25% | ~1.5× | Good | Minimal RAM environments |

### Export Models

```bash
# Export FP16 for GPU deployment
python exporter.py --model ./runs/lionai/final --format fp16

# Export INT8 for CPU deployment
python exporter.py --model ./runs/lionai/final --format int8

# Export all formats with benchmarks
python exporter.py --model ./runs/lionai/final --format all --benchmark

# Generate model card
python exporter.py --model ./runs/lionai/final --card
```

### Compress Checkpoints

```bash
python exporter.py --model ./runs/lionai/final --format compressed
```

---

## Hardware Requirements

### Minimum
- CPU: 2 cores (x86_64 or ARM)
- RAM: 4 GB
- Storage: 2 GB
- OS: Windows 10 / macOS 10.14 / Ubuntu 18.04

### Recommended
- CPU: 4+ cores
- RAM: 8 GB
- Storage: 10 GB SSD
- OS: Latest stable

### GPU Support
- **NVIDIA**: CUDA 11.8+ (any VRAM ≥ 4 GB)
- **Apple Silicon**: MPS backend (M1/M2/M3 — 8 GB unified memory)
- **AMD**: ROCm builds of PyTorch (experimental)

### Model/RAM Guide

| Model | Quantization | Min RAM |
|---|---|---|
| small (50M) | FP32 | 4 GB |
| small (50M) | INT8 | 2 GB |
| medium (125M) | FP32 | 4 GB |
| medium (125M) | INT8 | 4 GB |
| large (350M) | FP32 | 8 GB |
| large (350M) | INT8 | 6 GB |
| large (350M) | INT4 | 4 GB |

---

## Benchmarks

_Results from the built-in benchmark suite (`python evaluate.py`).
Values are illustrative targets — actual results depend on training data and duration._

### Speed (tokens/second, medium model)

| Device | FP32 | INT8 | INT4 |
|---|---|---|---|
| CPU (4-core) | 15–30 | 25–50 | 20–40 |
| CPU (8-core) | 25–50 | 45–90 | 35–70 |
| NVIDIA GPU (4GB) | 150–300 | 250–500 | 200–400 |
| Apple M2 (MPS) | 80–160 | N/A | N/A |

### Running the Benchmark Suite

```bash
python evaluate.py \
    --model ./runs/lionai/final \
    --device cpu \
    --output ./benchmark_results.json
```

---

## Project Structure

```
lionai/
├── model.py              Core transformer architecture
├── tokenizer.py          BPE tokenizer implementation
├── train.py              Training engine and loop
├── chatbot.py            Interactive chat interface
├── memory.py             Three-layer memory system
├── knowledge.py          RAG knowledge engine
├── dataset_processor.py  Data ingestion and cleaning
├── tokenizer_trainer.py  Tokenizer training CLI
├── optimization.py       Quantization and compression
├── evaluate.py           Benchmarking and metrics
├── exporter.py           Multi-format model export
├── config.py             Configuration and security
├── demo_setup.py         Demo installation helper
├── requirements.txt      Python dependencies
├── README.md             This file
└── LICENSE.md            Proprietary license
```

Total: **15 files** (within the 20-file limit).

---

## Roadmap

### v1.0 — Current
- [x] Custom BPE tokenizer
- [x] Transformer model (GQA + RoPE + SwiGLU)
- [x] Full training loop
- [x] Three-layer memory system
- [x] RAG knowledge engine
- [x] Interactive chat interface
- [x] INT8/INT4 quantization
- [x] ONNX export
- [x] Security layer

### v1.1 — Near Term
- [ ] Web UI (local Flask/FastAPI server)
- [ ] Streaming HTTP API
- [ ] Fine-tuning on custom instruction data
- [ ] LoRA adapter support
- [ ] Plugin system for tool integration
- [ ] Improved tokenizer (BPE with character normalization)

### v1.2 — Medium Term
- [ ] Function/tool calling agent
- [ ] Browser automation integration
- [ ] Code interpreter tool
- [ ] Improved RAG with dense retrieval
- [ ] Speech-to-text input (local Whisper)
- [ ] Text-to-speech output

### v2.0 — Long Term
- [ ] Vision encoder (local image understanding)
- [ ] Multimodal architecture
- [ ] Speculative decoding for faster inference
- [ ] Flash Attention 2 integration
- [ ] WebAssembly build for browser deployment
- [ ] Android/iOS packaging
- [ ] Federated learning support

---

## License

This software is proprietary and confidential. All rights reserved.
See [LICENSE.md](LICENSE.md) for the complete terms.

**Permitted:** Personal use on your own device.

**Not permitted:** Redistribution, modification, commercial use, public hosting, derivative models, code copying, reverse engineering.
