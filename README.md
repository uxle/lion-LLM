# 🦁 LionAI — Lion LLM (LLLM)
### Next-Generation Local AI with Real-Time Learning

**Fully offline · Self-hosted · Learns from every conversation · No cloud APIs**

---

## What Makes LionAI Different

Most local AI systems are static — they generate text but never improve. LionAI is different:

| Feature | LionAI | Typical local LLM |
|---|---|---|
| Learns from chat in real-time | ✅ LoRA micro-updates | ❌ Static weights |
| Understands intent | ✅ Intent classifier | ❌ No |
| Chain-of-thought reasoning | ✅ Built-in CoT | ❌ Prompt-only |
| Self-verifies responses | ✅ Quality checker | ❌ No |
| Explicit correction learning | ✅ /correct command | ❌ No |
| Remembers across sessions | ✅ Multi-tier memory | ❌ No |
| Works on AMD/Intel CPU | ✅ Fully tested | ⚠️ Often broken |

---

## Quick Start (your hardware: i5-10th + RX550 + 16GB RAM)

```bash
pip install torch

# 1. Setup (creates micro model, auto-sizes vocab + seq_len)
python demo_setup.py

# 2. Train on your data
python train.py --dataset ./data/train.jsonl
# Auto-detects: micro model, vocab=150, seq=32, ~100 steps

# 3. Chat with real-time learning
python chatbot.py --model ./runs/lionai/final
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

### How it works under the hood

```
Each turn:
  1. Score response quality (length, coherence, novelty, safety)
  2. Store (prompt, response, reward) in learner database
  3. Every 4 turns: micro-gradient step on LoRA adapters
  4. Every 20 turns: replay top-reward past turns
  5. EWC penalty prevents forgetting old knowledge
  6. Contrastive step when /correct is used
```

---

## Reasoning Pipeline

Every complex query goes through:

```
User query → Intent Classification → Chain-of-Thought → Generation → Self-Verification
```

### Intent Types Detected

- `question_factual` — definitions, facts, explanations
- `question_how` — step-by-step instructions
- `question_why` — cause and effect reasoning
- `task_code` — write/fix/debug code
- `task_math` — calculations with shown work
- `task_analyse` — structured analysis
- `correction` — user correcting the AI
- `feedback_positive/negative` — sentiment detection
- `memory_store/query` — remember/recall commands
- `conversation` — casual chat

### Chain-of-Thought Example

```
You: Why does Python use indentation?

[intent: question_why | steps: 3 | 0.8ms]

LionAI: Python uses indentation because...
  OBSERVE: query_why type | entities: python, indentation
  REASON:  Explanation query — will provide cause-and-effect
  PLAN:    1. State direct answer → 2. Explain reasoning → 3. Give example
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      LionAI System                          │
├──────────────┬──────────────┬───────────────────────────────┤
│  REASONING   │   LEARNING   │        GENERATION             │
│              │              │                               │
│ IntentClf    │ OnlineLearner│ GQA Attention (Flash)         │
│ ChainOfThought│ RewardEst.  │ SwiGLU FFN                    │
│ SelfVerifier │ EWCPenalty   │ RoPE Embeddings               │
│ ConfidenceEst│ ExpReplay    │ KV-Cache (fp16)               │
│ EntityExtract│ ContrastLoss │ Top-k/p/min-p sampling        │
├──────────────┴──────────────┴───────────────────────────────┤
│                   MEMORY SYSTEM                             │
│  Short-term (context) │ Long-term (SQLite+BM25) │ Semantic  │
├─────────────────────────────────────────────────────────────┤
│                 KNOWLEDGE ENGINE (RAG)                      │
│  Hybrid BM25 + FTS5 retrieval │ SimHash dedup               │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Guide

### Your Setup (i5-10th + RX550 4GB + 16GB RAM)

| Model | RAM | Speed | Best for |
|---|---|---|---|
| micro (15M) | ~60 MB | Fast | Testing, small datasets |
| small (50M) | ~200 MB | Good | Personal assistant |
| medium (125M) | ~500 MB | Slower | Better quality |

**Recommended for your hardware:**
```bash
python train.py --model-size small --vocab 2000
python chatbot.py --model ./runs/lionai/final --quantize int8
```

### AMD RX550 Note

The RX550 uses ROCm (not CUDA). LionAI auto-detects this.
If PyTorch-ROCm is installed: `device=cuda` with AMD detection.
If not: falls back to CPU (still fast with i5-10th + all cores used).

**Install PyTorch-ROCm (optional, for GPU acceleration):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## Training Guide

### For small datasets (your use case)

LionAI auto-configures everything based on your dataset size:

```bash
# Process your text files
python dataset_processor.py --sources ./mydata/ --output ./data

# Train (fully auto-configured)
python train.py --dataset ./data/train.jsonl

# What gets auto-selected:
#   50 words  → vocab=150,  seq=25,  steps=100
#   500 words → vocab=500,  seq=64,  steps=500
#   5000 words → vocab=2000, seq=128, steps=2000
```

### Manual control

```bash
python train.py \
  --dataset ./data/train.jsonl \
  --model-size micro \
  --vocab 512 \
  --seq-len 64 \
  --steps 500 \
  --batch 4
```

---

## All Commands

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

### Settings Commands

| Command | Description |
|---|---|
| `/stats` | System statistics |
| `/hardware` | Hardware profile |
| `/config KEY=VAL` | Change settings |
| `/mode sample\|contrastive\|beam` | Generation mode |
| `/quant none\|int8\|int4` | Change quantization |
| `/system TEMPLATE` | Set system prompt template |

### Config Keys

| Key | Default | Description |
|---|---|---|
| `temp` | 0.8 | Temperature (creativity) |
| `top_k` | 40 | Top-k sampling |
| `top_p` | 0.92 | Nucleus sampling |
| `max_tokens` | 256 | Max response length |
| `reasoning` | true | Chain-of-thought on/off |
| `learn` | true | Auto-learning on/off |
| `intent` | true | Show intent detection |
| `verify` | true | Self-verification on/off |

---

## Project Files

| File | Purpose |
|---|---|
| `model.py` | LionLLM transformer architecture |
| `tokenizer.py` | BPE tokenizer with incremental training |
| `train.py` | Training pipeline (auto-configured) |
| `chatbot.py` | Interactive chat interface |
| `learner.py` | **Real-time LoRA online learning** |
| `reasoner.py` | **Chain-of-thought + intent + verification** |
| `memory.py` | Three-tier memory system |
| `knowledge.py` | RAG knowledge engine |
| `optimization.py` | INT4/INT8/LoRA/pruning |
| `config.py` | Hardware detection + system config |
| `dataset_processor.py` | Data ingestion + cleaning |
| `tokenizer_trainer.py` | Tokenizer training CLI |
| `evaluate.py` | Benchmarking + quality metrics |
| `exporter.py` | Multi-format model export |
| `demo_setup.py` | Quick installation helper |

---

## License

Proprietary — All Rights Reserved. See `LICENSE.md`.
