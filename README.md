# 🦁 LionAI — Lion LLM (LLLM)
### Next-Generation Local AI with Real-Time Learning

**Fully offline · Self-hosted · Learns from every conversation · SafeTensors · No cloud APIs**

---

## What Makes LionAI Different

Most local AI systems are static — they generate text but never improve. LionAI is different:

| Feature | LionAI | Typical local LLM |
|---|---|---|
| **Real-time Chat Learning** | ✅ LoRA micro-updates | ❌ Static weights |
| **Dynamic Plan Verification** | ✅ Intent classifier + plan refiner | ❌ Prompt-only / no checks |
| **Self-verifying Responses** | ✅ Quality checker & NaNs scan | ❌ No checks |
| **RRF SQLite Search** | ✅ SQLite FTS5 + BM25 fusion | ❌ Simple text matching |
| **Memory / Forget Control** | ✅ Fuzzy key forget + category retrieval | ❌ No memory |
| **SafeTensors Export** | ✅ Native (numpy/pickle-free) SafeTensors | ❌ Pickle files (.bin/.pt) only |
| **Prunable Vocabulary** | ✅ Post-training BPE tokenizer compaction | ❌ Static vocabulary |
| **Hardware Auto-Tuning** | ✅ Auto-tuned host `.env` configuration | ❌ Manual config only |

---

## Quick Start (your hardware: i5-10th + RX550 + 16GB RAM)

```bash
# 1. Setup (creates micro model, auto-detects resources, writes tailored .env)
python demo_setup.py

# 2. Train on your data
python train.py --dataset ./data/train.jsonl
# Auto-detects: micro model, vocab=512, seq=64, steps=500

# 3. Chat with real-time learning
python chatbot.py --model ./runs/lionai/final
```

---

## New Capabilities & Utilities

### 1. SafeTensors Model Exports
LionAI supports exporting weights to the safe and modern `SafeTensors` format (`model.safetensors`), preventing pickle security vulnerabilities.
* **Pure Python/PyTorch Fallback**: Serializer works without hard dependencies on `numpy` or external libraries.
* **Cross-Format Equivalency Check**: The exporter verifies top-logit overlap, generation samples, and checks for NaN issues against the FP32 baseline automatically:
  ```bash
  python exporter.py --model ./runs/lionai/final --format safetensors --validate
  ```

### 2. Tokenizer BPE Pruning & Padding Alignment
* **BPE Vocabulary Compacting**: Remaps and prunes unused BPE tokens from trained tokenizers based on a specific text corpus, optimizing model embedding tables:
  ```bash
  python tokenizer_trainer.py prune --tokenizer ./runs/lionai/final --corpus ./data/train_chat.jsonl --output ./runs/lionai/pruned_tokenizer
  ```
* **Padding Alignment**: Supports `padding_side="left"|"right"` and `pad_to_multiple_of` (e.g. padding to multiples of 8 for Tensor Core optimization) for batch tokenization.
* **Thread-Pool Parallel Decoding**: Optimizes batch decoding speed across all available CPU cores.

### 3. BLEU / ROUGE-L Reference Evaluation
Evaluate generated model responses against reference ground-truth datasets to compute similarity scores (BLEU and ROUGE-L) without external library requirements:
```bash
python evaluate.py --model ./runs/lionai/final --reference-dataset ./data/train_chat.jsonl
```

### 4. Terminal ASCII Latency Distribution
Plots a latency distribution histogram directly in the terminal console when evaluating texts to visualize P50, P90, P95, and P99 latency bounds:
```bash
python evaluate.py --model ./runs/lionai/final --speed-only
```

---

## Real-Time Learning

LionAI learns from every conversation using **LoRA** (Low-Rank Adaptation).
Only 0.5–2% of parameters are updated per turn — the rest stay frozen.

### Teaching LionAI

```
You: What is the capital of Australia?
LionAI: The capital of Australia is Sydney.

/bad                          ← tell it that was wrong
/correct "Sydney" "Canberra"  ← teach the right answer

You: What is the capital of Australia?
LionAI: The capital of Australia is Canberra.  ← learned!
```

### Learning Commands

| Command | What it does |
|---|---|
| `/good` | Positive signal — reinforce this response style |
| `/bad` | Negative signal — avoid this response style |
| `/correct "WRONG" "RIGHT"` | Contrastive learning — teaches exact correction |
| `/learn_stats` | See how many updates have happened |
| `/save_lora` | Save learned weights to disk |

---

## Reasoning & Intent Pipeline

Every complex query goes through the reasoning orchestrator:

```
User query → Intent Classification → Plan Verification → Chain-of-Thought → Generation → Self-Verification
```

### Dynamic Plan Verification & Refinement
The `reasoner.py` planner checks if the generated reasoning steps align with the intent category. If key steps are missing (e.g., implementation targets in `task_code` or formula resolution steps in `task_math`), it automatically refines the plan and logs an informational warning.

---

## Hardware Guide

### Recommended Settings for i5-10th + RX550 + 16GB RAM
The `demo_setup.py` profile auto-selects and tunes these recommended parameters into the generated `.env` configuration:
```ini
LIONAI_DEVICE=cpu
LIONAI_QUANTIZATION=int8
LIONAI_TORCH_THREADS=8
LIONAI_MAX_NEW_TOKENS=512
LIONAI_TEMPERATURE=0.7
```

---

## All Chat Commands

### Chat Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/reset` | Clear conversation |
| `/save [name]` | Save session |
| `/load [name]` | Load session |
| `/export [file]` | Export conversation |

### Learning Commands

| Command | Description |
|---|---|
| `/good` | Mark last response as good ✓ |
| `/bad` | Mark last response as bad ✗ |
| `/correct "WRONG" "RIGHT"` | Teach correct answer |
| `/learn_stats` | Show learning progress |
| `/save_lora` | Save LoRA weights |

### Memory Commands

| Command | Description |
|---|---|
| `/memory` | List stored memories |
| `/learn KEY VALUE` | Store a fact |
| `/forget KEY` | Delete a fact |

### Knowledge Commands

| Command | Description |
|---|---|
| `/docs [path]` | Ingest document(s) |
| `/search QUERY` | Search knowledge base |

---

## Project Files

| File | Purpose |
|---|---|
| `model.py` | LionLLM transformer architecture withcos/sin & GQA mask caching |
| `tokenizer.py` | BPE tokenizer supporting parallel batch encode/decode and padding sides |
| `train.py` | Training pipeline (auto-configured) |
| `chatbot.py` | Interactive chat interface with unix-readline history |
| `learner.py` | **Real-time LoRA online learning** |
| `reasoner.py` | **Chain-of-thought + dynamic plan verification** |
| `memory.py` | Three-tier memory system with fuzzy delete options |
| `knowledge.py` | SQLite RAG knowledge engine combining FTS5 + BM25 (RRF) |
| `optimization.py` | INT4/INT8/LoRA/pruning |
| `config.py` | Hardware detection + system config validator |
| `dataset_processor.py` | Parallel data ingestion + cleaning |
| `tokenizer_trainer.py` | Tokenizer training and pruning CLI |
| `evaluate.py` | Benchmarking + BLEU/ROUGE-L similarity + ASCII plots |
| `exporter.py` | Multi-format model exporter & cross-format validation |
| `demo_setup.py` | Quick installation & auto-tuned environment builder |

---

## License

Proprietary — All Rights Reserved. See `LICENSE.md`.
