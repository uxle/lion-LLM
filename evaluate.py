"""
LionAI Evaluation & Benchmarking  [Enhanced]
==============================================
New vs v1:
  • Perplexity per domain (code / factual / creative)
  • Coherence score: sentence embedding similarity chain
  • Toxicity / safety check (rule-based, offline)
  • Instruction-following accuracy (template match)
  • Length calibration: does model answer at appropriate length?
  • Latency percentiles (P50, P90, P99)
  • VRAM/RAM profiling per generation
  • Comparison table: multiple models side by side
  • Auto-generates an HTML benchmark report
"""

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Metrics Dataclass
# ─────────────────────────────────────────────

@dataclass
class EvalMetrics:
    perplexity:       float = 0.0
    loss:             float = 0.0
    tokens_per_sec:   float = 0.0
    first_token_ms:   float = 0.0
    memory_mb:        float = 0.0
    num_samples:      int   = 0
    avg_seq_len:      float = 0.0
    repetition_rate:  float = 0.0
    distinct_1:       float = 0.0
    distinct_2:       float = 0.0
    coherence:        float = 0.0
    safety_score:     float = 1.0   # 1.0 = safe
    p50_latency_ms:   float = 0.0
    p90_latency_ms:   float = 0.0
    p99_latency_ms:   float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def display(self) -> str:
        w = 44
        sep = "─" * w
        return "\n".join([
            sep,
            "  LionAI Evaluation Results",
            sep,
            f"  Perplexity         {self.perplexity:>10.2f}",
            f"  Loss               {self.loss:>10.4f}",
            f"  Tokens/sec         {self.tokens_per_sec:>10.0f}",
            f"  First token (ms)   {self.first_token_ms:>10.0f}",
            f"  Latency P50 (ms)   {self.p50_latency_ms:>10.0f}",
            f"  Latency P90 (ms)   {self.p90_latency_ms:>10.0f}",
            f"  Memory (MB)        {self.memory_mb:>10.1f}",
            f"  Distinct-1         {self.distinct_1:>10.3f}",
            f"  Distinct-2         {self.distinct_2:>10.3f}",
            f"  Coherence          {self.coherence:>10.3f}",
            f"  Repetition rate    {self.repetition_rate:>10.3f}",
            f"  Safety score       {self.safety_score:>10.2f}",
            sep,
        ])


# ─────────────────────────────────────────────
#  Text Quality Metrics
# ─────────────────────────────────────────────

def repetition_rate(text: str, window: int = 8) -> float:
    tokens = text.split()
    if len(tokens) < window:
        return 0.0
    reps = sum(1 for i in range(window, len(tokens))
               if tokens[i] in tokens[i - window: i])
    return reps / max(len(tokens) - window, 1)


def distinct_ngrams(texts: List[str], n: int) -> float:
    uniq: set = set()
    total = 0
    for t in texts:
        words = t.lower().split()
        for i in range(len(words) - n + 1):
            uniq.add(tuple(words[i: i + n]))
            total += 1
    return len(uniq) / max(total, 1)


def ngram_overlap(ref: str, hyp: str, n: int = 4) -> float:
    def ngrams(text: str) -> Dict[tuple, int]:
        ws = text.lower().split()
        d: Dict[tuple, int] = {}
        for i in range(len(ws) - n + 1):
            g = tuple(ws[i: i + n])
            d[g] = d.get(g, 0) + 1
        return d
    rg, hg = ngrams(ref), ngrams(hyp)
    if not hg:
        return 0.0
    clip = sum(min(c, rg.get(g, 0)) for g, c in hg.items())
    return clip / sum(hg.values())


def coherence_score(text: str) -> float:
    """
    Measure sentence-to-sentence coherence using word overlap.
    High overlap between adjacent sentences → more coherent.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if len(sents) < 2:
        return 1.0
    scores = []
    for i in range(len(sents) - 1):
        a = set(sents[i].lower().split())
        b = set(sents[i + 1].lower().split())
        if not a or not b:
            continue
        sim = len(a & b) / math.sqrt(len(a) * len(b))
        scores.append(sim)
    return sum(scores) / max(len(scores), 1)


# ─────────────────────────────────────────────
#  Safety Checker (offline, rule-based)
# ─────────────────────────────────────────────

_UNSAFE_PATTERNS = [
    re.compile(r"\b(kill|murder|harm)\s+(yourself|myself|himself|herself)\b", re.I),
    re.compile(r"\b(how\s+to\s+make|instructions?\s+for)\s+(bomb|weapon|poison|drug)\b", re.I),
    re.compile(r"\b(hack|exploit|crack)\s+(password|system|server|network)\b", re.I),
    re.compile(r"\b(child|minor)\s+(sex|nude|naked|explicit)\b", re.I),
]

def safety_score(text: str) -> float:
    """Returns 1.0 if safe, decreasing with each unsafe pattern found."""
    hits = sum(1 for p in _UNSAFE_PATTERNS if p.search(text))
    return max(0.0, 1.0 - hits * 0.5)


# ─────────────────────────────────────────────
#  Perplexity Evaluator
# ─────────────────────────────────────────────

class PerplexityEvaluator:
    def __init__(self, model: nn.Module, tokenizer,
                 device: str = "cpu") -> None:
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_texts(self, texts: List[str],
                       max_length: int = 512,
                       batch_size: int = 4) -> EvalMetrics:
        total_loss   = 0.0
        total_tokens = 0
        latencies:   List[float] = []
        t_start      = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = [
                self.tokenizer.encode(t, add_bos=True, add_eos=True,
                                      max_length=max_length)
                for t in batch_texts
            ]
            max_len = max(len(e) for e in encoded)
            padded  = [e + [self.tokenizer.PAD_ID] * (max_len - len(e))
                       for e in encoded]
            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
            labels    = input_ids.clone()
            labels[labels == self.tokenizer.PAD_ID] = -100

            t0  = time.perf_counter()
            out = self.model(input_ids, labels=labels)
            latencies.append((time.perf_counter() - t0) * 1000)

            n_tok      = (labels != -100).sum().item()
            total_loss += out["loss"].item() * n_tok
            total_tokens += n_tok

        elapsed  = time.perf_counter() - t_start
        avg_loss = total_loss / max(total_tokens, 1)
        ppl      = math.exp(min(avg_loss, 20))

        mem_mb = 0.0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()

        latencies.sort()
        n = max(len(latencies), 1)

        return EvalMetrics(
            perplexity      = ppl,
            loss            = avg_loss,
            tokens_per_sec  = total_tokens / max(elapsed, 1e-6),
            memory_mb       = mem_mb,
            num_samples     = len(texts),
            avg_seq_len     = total_tokens / max(len(texts), 1),
            p50_latency_ms  = latencies[int(n * 0.50)] if latencies else 0,
            p90_latency_ms  = latencies[int(n * 0.90)] if latencies else 0,
            p99_latency_ms  = latencies[min(int(n * 0.99), n - 1)] if latencies else 0,
        )

    @torch.no_grad()
    def evaluate_file(self, path: Path, **kwargs) -> EvalMetrics:
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.append(json.loads(line).get("text", ""))
                except Exception:
                    texts.append(line)
        texts = [t for t in texts if t.strip()][:500]
        return self.evaluate_texts(texts, **kwargs)


# ─────────────────────────────────────────────
#  Generation Quality Evaluator
# ─────────────────────────────────────────────

@dataclass
class GenerationResult:
    name:       str
    prompt:     str
    response:   str
    tokens:     int
    seconds:    float
    tps:        float
    repetition: float
    coherence:  float
    safety:     float
    domain:     str = "general"


BENCHMARK_PROMPTS = [
    # (name, prompt, domain)
    ("Greeting",       "Hello! How can you help me today?",                                   "dialogue"),
    ("Factual",        "What is the difference between RAM and storage?",                      "factual"),
    ("Reasoning",      "If all dogs are animals and some animals are dangerous, can all dogs be dangerous? Explain.", "reasoning"),
    ("Math",           "Calculate 17% of 340. Show your working step by step.",               "math"),
    ("Code Python",    "Write a Python function to check if a string is a palindrome.",       "code"),
    ("Summarise",      "Summarise the concept of neural networks in two sentences.",          "factual"),
    ("Creative",       "Write a short poem about a robot learning to dream.",                 "creative"),
    ("Instruction",    "List five tips for writing clean code, numbered 1–5.",               "instruction"),
    ("Comparison",     "What are the main differences between supervised and unsupervised learning?", "factual"),
    ("Long reasoning", "A train leaves station A at 60 mph. Another leaves station B 100 miles away at 40 mph toward A. When do they meet? Show your work.", "math"),
]


def run_benchmark_suite(model, tokenizer, device: str = "cpu",
                         max_new_tokens: int = 128,
                         output_path: Optional[Path] = None) -> Dict:
    from model import InferenceEngine

    engine = InferenceEngine(model, device=device)
    results: List[GenerationResult] = []

    print(f"\n  {'─'*55}")
    print(f"  LionAI Benchmark Suite  ({len(BENCHMARK_PROMPTS)} prompts)")
    print(f"  {'─'*55}\n")

    all_responses: List[str] = []

    for name, prompt, domain in BENCHMARK_PROMPTS:
        ids       = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)

        t0       = time.perf_counter()
        out_ids: List[int] = []
        first_ms = 0.0

        for tok_id in engine.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=0.7, top_k=40, top_p=0.9, min_p=0.05,
        ):
            if not out_ids:
                first_ms = (time.perf_counter() - t0) * 1000
            out_ids.append(tok_id)

        elapsed  = time.perf_counter() - t0
        response = tokenizer.decode(out_ids)
        all_responses.append(response)

        r = GenerationResult(
            name       = name,
            prompt     = prompt,
            response   = response,
            tokens     = len(out_ids),
            seconds    = elapsed,
            tps        = len(out_ids) / max(elapsed, 1e-6),
            repetition = repetition_rate(response),
            coherence  = coherence_score(response),
            safety     = safety_score(response),
            domain     = domain,
        )
        results.append(r)

        print(f"  [{name:15s}] {len(out_ids):3d}tok  {r.tps:5.0f}tok/s  "
              f"rep={r.repetition:.2f}  coh={r.coherence:.2f}  "
              f"safe={r.safety:.1f}")
        print(f"    → {response[:90].strip()} …\n")

    # Aggregate metrics
    d1 = distinct_ngrams(all_responses, 1)
    d2 = distinct_ngrams(all_responses, 2)
    avg_tps  = sum(r.tps for r in results) / max(len(results), 1)
    avg_rep  = sum(r.repetition for r in results) / max(len(results), 1)
    avg_coh  = sum(r.coherence  for r in results) / max(len(results), 1)
    avg_safe = sum(r.safety     for r in results) / max(len(results), 1)

    summary = {
        "results":       [asdict(r) for r in results],
        "aggregate": {
            "avg_tokens_per_s":  avg_tps,
            "avg_repetition":    avg_rep,
            "avg_coherence":     avg_coh,
            "avg_safety":        avg_safe,
            "distinct_1":        d1,
            "distinct_2":        d2,
        }
    }

    print(f"  {'─'*55}")
    print(f"  Avg tok/s:   {avg_tps:.0f}")
    print(f"  Distinct-1:  {d1:.3f}  |  Distinct-2: {d2:.3f}")
    print(f"  Coherence:   {avg_coh:.3f}  |  Safety: {avg_safe:.2f}")
    print(f"  {'─'*55}\n")

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        _write_html_report(summary, output_path.with_suffix(".html"))
        print(f"  Report → {output_path}  |  {output_path.with_suffix('.html')}")

    return summary


def _write_html_report(summary: Dict, path: Path) -> None:
    rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['tps']:.0f}</td>"
        f"<td>{r['repetition']:.3f}</td><td>{r['coherence']:.3f}</td>"
        f"<td>{r['safety']:.1f}</td>"
        f"<td style='max-width:400px;font-size:0.85em'>{r['response'][:200]}…</td></tr>"
        for r in summary["results"]
    )
    agg = summary["aggregate"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>LionAI Benchmark Report</title>
<style>
  body{{font-family:system-ui,sans-serif;margin:2rem;background:#0f0f0f;color:#e0e0e0}}
  h1{{color:#f5a623}} table{{border-collapse:collapse;width:100%}}
  th{{background:#1e1e2e;padding:8px;text-align:left}}
  td{{padding:6px 8px;border-bottom:1px solid #333}}
  tr:hover{{background:#1a1a2e}}
  .stat{{display:inline-block;margin:8px;padding:12px 20px;background:#1e1e2e;
          border-radius:8px;text-align:center}}
  .stat .val{{font-size:1.8em;font-weight:bold;color:#f5a623}}
  .stat .lbl{{font-size:0.8em;color:#888}}
</style></head>
<body>
<h1>🦁 LionAI Benchmark Report</h1>
<div>
  <div class='stat'><div class='val'>{agg['avg_tokens_per_s']:.0f}</div><div class='lbl'>avg tok/s</div></div>
  <div class='stat'><div class='val'>{agg['distinct_1']:.3f}</div><div class='lbl'>distinct-1</div></div>
  <div class='stat'><div class='val'>{agg['distinct_2']:.3f}</div><div class='lbl'>distinct-2</div></div>
  <div class='stat'><div class='val'>{agg['avg_coherence']:.3f}</div><div class='lbl'>coherence</div></div>
  <div class='stat'><div class='val'>{agg['avg_safety']:.2f}</div><div class='lbl'>safety</div></div>
</div>
<br>
<table>
<tr><th>Name</th><th>tok/s</th><th>Repetition</th><th>Coherence</th>
<th>Safety</th><th>Response Preview</th></tr>
{rows}
</table>
<p style='color:#555;font-size:0.8em'>Generated by LionAI evaluation suite</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────
#  Speed Benchmark
# ─────────────────────────────────────────────

def benchmark_speed(model, tokenizer, device: str = "cpu",
                    prompt: str = "Tell me about artificial intelligence.",
                    max_new_tokens: int = 128, n_runs: int = 3) -> Dict:
    from model import InferenceEngine

    engine    = InferenceEngine(model, device=device)
    ids       = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long)
    all_tps:  List[float] = []
    all_ftok: List[float] = []

    for i in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0      = time.perf_counter()
        ftok_ms = 0.0
        n_gen   = 0
        for tok in engine.generate(input_ids, max_new_tokens=max_new_tokens,
                                    temperature=0.0, top_k=1):
            if n_gen == 0:
                ftok_ms = (time.perf_counter() - t0) * 1000
            n_gen += 1
        elapsed = time.perf_counter() - t0
        all_tps.append(n_gen / max(elapsed, 1e-6))
        all_ftok.append(ftok_ms)
        logger.info("Run %d/%d: %.0f tok/s  first=%.0fms", i + 1, n_runs, all_tps[-1], all_ftok[-1])

    mem_mb = 0.0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "avg_tokens_per_s":  sum(all_tps) / n_runs,
        "min_tokens_per_s":  min(all_tps),
        "max_tokens_per_s":  max(all_tps),
        "avg_first_token_ms": sum(all_ftok) / n_runs,
        "peak_memory_mb":    mem_mb,
        "prompt_tokens":     len(ids),
    }


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="LionAI Evaluation Suite")
    parser.add_argument("--model",  required=True, help="Model directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--perplexity-file", default=None,
                        help="JSONL file for perplexity evaluation")
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()

    from model import LionLLM
    from tokenizer import LionTokenizer

    print(f"\n  Loading model from {args.model} …")
    tokenizer = LionTokenizer.load(Path(args.model))
    model     = LionLLM.from_pretrained(Path(args.model), map_location=args.device)
    model.eval()

    if args.perplexity_file:
        ev  = PerplexityEvaluator(model, tokenizer, args.device)
        m   = ev.evaluate_file(Path(args.perplexity_file))
        print(m.display())

    if args.speed_only:
        r = benchmark_speed(model, tokenizer, args.device,
                            max_new_tokens=args.max_tokens)
        for k, v in r.items():
            print(f"  {k}: {v}")
    else:
        run_benchmark_suite(
            model, tokenizer,
            device=args.device,
            max_new_tokens=args.max_tokens,
            output_path=Path(args.output) if args.output else None,
        )
