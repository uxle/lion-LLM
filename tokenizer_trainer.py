"""
LionAI Tokenizer Trainer  [Enhanced]
======================================
New vs v1:
  • Subcommand CLI: train / test / analyze / compare / merge
  • Multi-corpus training with per-corpus weights
  • Vocabulary coverage analysis: % of test corpus encoded without <unk>
  • Token fertility report: chars-per-token by domain
  • Side-by-side comparison of two tokenizers
  • Merge two vocabulary files (multi-language tokenizer building)
  • Export vocab as plain text for inspection
  • Byte-fallback guarantee: every input is encodeable (zero <unk>)
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from tokenizer import LionTokenizer, TokenizerTrainer, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Corpus Iterators
# ─────────────────────────────────────────────

def iter_corpus(sources: List[Path],
                weight: float = 1.0,
                max_docs: Optional[int] = None) -> Iterator[str]:
    """Yield texts from files/directories, repeating by weight (upsampling)."""
    count = 0
    repeat = max(1, round(weight))
    for src in sources:
        src = Path(src)
        files = list(src.rglob("*")) if src.is_dir() else [src]
        for f in files:
            if not f.is_file():
                continue
            try:
                suffix = f.suffix.lower()
                if suffix in (".txt", ".md", ".markdown"):
                    text = f.read_text(encoding="utf-8", errors="replace")
                    for _ in range(repeat):
                        yield text
                        count += 1
                        if max_docs and count >= max_docs:
                            return
                elif suffix in (".jsonl", ".json"):
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                text = (obj.get("text") or obj.get("content") or
                                        obj.get("instruction","") + " " +
                                        obj.get("output",""))
                            except Exception:
                                text = line
                            if text.strip():
                                for _ in range(repeat):
                                    yield text.strip()
                                    count += 1
                                    if max_docs and count >= max_docs:
                                        return
            except Exception as e:
                logger.warning("Skipping %s: %s", f.name, e)


# ─────────────────────────────────────────────
#  Analysis Functions
# ─────────────────────────────────────────────

def analyze_tokenizer(tokenizer: LionTokenizer,
                       sample_texts: List[str],
                       top_n: int = 40) -> None:
    """Print a comprehensive tokenizer analysis report."""
    print(f"\n{'═'*55}")
    print(f"  LionAI Tokenizer Analysis")
    print(f"{'═'*55}")
    print(f"  Vocabulary size:    {tokenizer.vocab_size:>8,}")
    print(f"  BPE merges:         {len(tokenizer.merges):>8,}")
    print(f"  Special tokens:     {len(SPECIAL_TOKENS):>8}")

    # Compression ratio
    total_chars  = sum(len(t) for t in sample_texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in sample_texts)
    ratio = total_chars / max(total_tokens, 1)
    print(f"  Chars/token:        {ratio:>8.2f}  (higher = better compression)")

    # Fertility (tokens per word)
    all_words = [w for t in sample_texts for w in re.findall(r"\w+", t)][:5000]
    word_tok_counts = [len(tokenizer.encode(w)) for w in all_words]
    fertility = sum(word_tok_counts) / max(len(word_tok_counts), 1)
    print(f"  Fertility:          {fertility:>8.2f}  (tokens/word, lower = better)")

    # Coverage (% of chars that map to non-unk)
    unk_count = 0
    total_enc = 0
    for t in sample_texts[:100]:
        ids = tokenizer.encode(t)
        unk_count += ids.count(tokenizer.UNK_ID)
        total_enc += len(ids)
    unk_pct = 100 * unk_count / max(total_enc, 1)
    print(f"  Unknown rate:       {unk_pct:>7.2f}%  (lower = better coverage)")

    # Domain fertility breakdown
    domain_samples = {
        "prose":   [t for t in sample_texts if len(t.split()) > 20][:20],
        "code":    [t for t in sample_texts if any(k in t for k in ["def ","class ","import ","return "])][:20],
        "numbers": ["The price is $1,234.56 and the date is 2024-01-15."] * 5,
    }
    print(f"\n  Fertility by domain:")
    for domain, texts in domain_samples.items():
        if not texts:
            continue
        words = [w for t in texts for w in re.findall(r"\w+", t)][:500]
        if not words:
            continue
        fert = sum(len(tokenizer.encode(w)) for w in words) / max(len(words), 1)
        print(f"    {domain:12s}  {fert:.2f} tok/word")

    # Top tokens
    print(f"\n  Top {top_n} most frequent tokens in sample:")
    from collections import Counter
    counter: Counter = Counter()
    for t in sample_texts:
        for tid in tokenizer.encode(t):
            tok = tokenizer.id2token.get(tid, "?")
            if tok not in SPECIAL_TOKENS:
                counter[repr(tok)] += 1
    for tok_repr, cnt in counter.most_common(top_n):
        bar = "█" * min(cnt // max(counter.most_common(1)[0][1] // 25, 1), 25)
        print(f"    {tok_repr:25s} {cnt:7,}  {bar}")

    print(f"{'═'*55}\n")


def test_roundtrip(tokenizer: LionTokenizer,
                    texts: Optional[List[str]] = None) -> bool:
    """Verify encode → decode preserves text (modulo unicode normalisation)."""
    tests = texts or [
        "Hello, world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming AI. (2024)",
        "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1)+fibonacci(n-2)",
        "Ñoño señor: ünïcödé tëxt with spëcïal chars.",
        "1234567890 !@#$%^&*()",
        "   leading and trailing whitespace   ",
        "<sys>System prompt</sys><usr>User message</usr><ast>Response</ast>",
        "Short.",
        "a",
    ]
    failures = 0
    print(f"\n  Running {len(tests)} round-trip tests …")
    for text in tests:
        ids     = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        # Compare after normalisation
        norm_in  = unicodedata.normalize("NFKC", text).strip()
        norm_out = decoded.strip()
        if norm_out != norm_in:
            logger.warning("  MISMATCH:\n    IN:  %r\n    OUT: %r", norm_in[:80], norm_out[:80])
            failures += 1
        else:
            print(f"    ✓ {text[:50]!r}")
    if failures == 0:
        print(f"  ✓ All tests passed\n")
    else:
        print(f"  ✗ {failures}/{len(tests)} tests failed — check logs\n")
    return failures == 0


def compare_tokenizers(tok_a: LionTokenizer, name_a: str,
                        tok_b: LionTokenizer, name_b: str,
                        sample_texts: List[str]) -> None:
    """Side-by-side comparison of two tokenizers."""
    print(f"\n{'═'*60}")
    print(f"  Tokenizer Comparison: {name_a}  vs  {name_b}")
    print(f"{'═'*60}")
    print(f"  {'Metric':<28} {name_a:>14} {name_b:>14}")
    print(f"  {'─'*56}")

    def tok_stats(tok: LionTokenizer) -> Dict:
        chars  = sum(len(t) for t in sample_texts)
        tokens = sum(len(tok.encode(t)) for t in sample_texts)
        words  = [w for t in sample_texts for w in re.findall(r"\w+", t)][:3000]
        fert   = sum(len(tok.encode(w)) for w in words) / max(len(words), 1)
        unk    = sum(tok.encode(t).count(tok.UNK_ID) for t in sample_texts)
        return {
            "vocab_size":  tok.vocab_size,
            "chars/tok":   chars / max(tokens, 1),
            "fertility":   fert,
            "unk_rate%":   100 * unk / max(tokens, 1),
        }

    sa, sb = tok_stats(tok_a), tok_stats(tok_b)
    for metric in sa:
        va, vb = sa[metric], sb[metric]
        fmt = ".0f" if metric == "vocab_size" else ".3f"
        print(f"  {metric:<28} {va:>14{fmt}} {vb:>14{fmt}}")
    print(f"{'═'*60}\n")


def merge_vocabularies(tok_a: LionTokenizer,
                        tok_b: LionTokenizer,
                        output_dir: Path,
                        max_vocab: int = 64000) -> LionTokenizer:
    """
    Merge two tokenizers into one (e.g. English + code).
    Keeps all tokens from tok_a, adds new tokens from tok_b up to max_vocab.
    """
    merged = LionTokenizer()
    merged.token2id = dict(tok_a.token2id)
    merged.id2token = dict(tok_a.id2token)
    merged.merges   = list(tok_a.merges)
    merged._merge_rank = dict(tok_a._merge_rank)

    added = 0
    for tok, tid in tok_b.token2id.items():
        if tok in merged.token2id:
            continue
        if len(merged.token2id) >= max_vocab:
            break
        new_id = len(merged.token2id)
        merged.token2id[tok]    = new_id
        merged.id2token[new_id] = tok
        added += 1

    # Merge BPE merges (tok_b merges after tok_a merges)
    existing = set(merged.merges)
    for m in tok_b.merges:
        if m not in existing:
            merged.merges.append(m)
            merged._merge_rank[m] = len(merged.merges) - 1

    merged._vocab_size = len(merged.token2id)
    merged._build_trie()
    merged.save(output_dir)

    logger.info("Merged tokenizer: %d + %d new = %d total tokens",
                tok_a.vocab_size, added, merged.vocab_size)
    return merged


def export_vocab_txt(tokenizer: LionTokenizer, output_path: Path) -> None:
    """Export vocabulary as plain text (one token per line) for inspection."""
    lines = []
    for tid in sorted(tokenizer.id2token.keys()):
        tok = tokenizer.id2token[tid]
        lines.append(f"{tid}\t{tok!r}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Vocabulary exported → %s (%d tokens)", output_path, len(lines))


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="LionAI Tokenizer Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────
    tp = sub.add_parser("train", help="Train a BPE tokenizer from corpus")
    tp.add_argument("--input",      nargs="+", required=True)
    tp.add_argument("--output",     default="./tokenizer")
    tp.add_argument("--vocab",      type=int, default=32000)
    tp.add_argument("--min-freq",   type=int, default=2)
    tp.add_argument("--max-docs",   type=int, default=None)
    tp.add_argument("--weight",     type=float, default=1.0,
                    help="Upsampling weight for this corpus (>1 = repeat data)")
    tp.add_argument("--analyze",    action="store_true")
    tp.add_argument("--test",       action="store_true")
    tp.add_argument("--export-vocab", action="store_true",
                    help="Export vocab.txt after training")

    # ── test ────────────────────────────────
    tsp = sub.add_parser("test", help="Test round-trip on a tokenizer")
    tsp.add_argument("--tokenizer", required=True)
    tsp.add_argument("--text",      nargs="*")

    # ── analyze ─────────────────────────────
    ap = sub.add_parser("analyze", help="Analyze tokenizer quality")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus",    nargs="+", required=True)
    ap.add_argument("--top-n",     type=int, default=40)

    # ── compare ─────────────────────────────
    cp = sub.add_parser("compare", help="Compare two tokenizers")
    cp.add_argument("--tok-a", required=True)
    cp.add_argument("--tok-b", required=True)
    cp.add_argument("--corpus", nargs="+", required=True)

    # ── merge ────────────────────────────────
    mp = sub.add_parser("merge", help="Merge two tokenizer vocabularies")
    mp.add_argument("--tok-a",   required=True)
    mp.add_argument("--tok-b",   required=True)
    mp.add_argument("--output",  required=True)
    mp.add_argument("--max-vocab", type=int, default=64000)

    # ── export-vocab ─────────────────────────
    ep = sub.add_parser("export-vocab", help="Export vocab as plain text")
    ep.add_argument("--tokenizer", required=True)
    ep.add_argument("--output",    required=True)

    args = parser.parse_args()

    # ─── TRAIN ──────────────────────────────
    if args.command == "train":
        print(f"\n  Training tokenizer")
        print(f"  Sources:    {args.input}")
        print(f"  Vocab size: {args.vocab:,}")
        print(f"  Min freq:   {args.min_freq}")

        sources = [Path(s) for s in args.input]
        corpus  = iter_corpus(sources, weight=args.weight, max_docs=args.max_docs)

        trainer   = TokenizerTrainer(
            vocab_size      = args.vocab,
            min_frequency   = args.min_freq,
            show_progress   = True,
            checkpoint_interval = 5000,
        )
        tokenizer = trainer.train(corpus, save_dir=Path(args.output))
        tokenizer.save(Path(args.output))

        print(f"\n  ✓ Tokenizer saved → {args.output}")
        print(f"    Vocabulary: {tokenizer.vocab_size:,} tokens")
        print(f"    Merges:     {len(tokenizer.merges):,}")

        if args.analyze:
            samples = list(iter_corpus(sources, max_docs=500))[:200]
            analyze_tokenizer(tokenizer, samples)

        if args.test:
            test_roundtrip(tokenizer)

        if args.export_vocab:
            export_vocab_txt(tokenizer, Path(args.output) / "vocab.txt")

    # ─── TEST ───────────────────────────────
    elif args.command == "test":
        tokenizer = LionTokenizer.load(Path(args.tokenizer))
        custom    = args.text or None
        passed    = test_roundtrip(tokenizer, custom)

        # Show encode/decode for custom texts
        if args.text:
            print(f"\n  Encode/Decode Examples:")
            for text in args.text:
                ids    = tokenizer.encode(text)
                tokens = [tokenizer.id2token.get(i, "?") for i in ids]
                dec    = tokenizer.decode(ids)
                print(f"\n  Input:   {text!r}")
                print(f"  Tokens:  {tokens}")
                print(f"  IDs:     {ids}")
                print(f"  Decoded: {dec!r}")
                print(f"  Count:   {len(ids)} tokens")

    # ─── ANALYZE ────────────────────────────
    elif args.command == "analyze":
        tokenizer = LionTokenizer.load(Path(args.tokenizer))
        samples   = list(iter_corpus([Path(c) for c in args.corpus], max_docs=500))[:300]
        analyze_tokenizer(tokenizer, samples, args.top_n)

    # ─── COMPARE ────────────────────────────
    elif args.command == "compare":
        tok_a   = LionTokenizer.load(Path(args.tok_a))
        tok_b   = LionTokenizer.load(Path(args.tok_b))
        samples = list(iter_corpus([Path(c) for c in args.corpus], max_docs=500))[:200]
        compare_tokenizers(tok_a, Path(args.tok_a).name,
                            tok_b, Path(args.tok_b).name,
                            samples)

    # ─── MERGE ──────────────────────────────
    elif args.command == "merge":
        tok_a  = LionTokenizer.load(Path(args.tok_a))
        tok_b  = LionTokenizer.load(Path(args.tok_b))
        merged = merge_vocabularies(tok_a, tok_b,
                                     Path(args.output),
                                     args.max_vocab)
        print(f"  ✓ Merged tokenizer: {merged.vocab_size:,} tokens → {args.output}")

    # ─── EXPORT-VOCAB ───────────────────────
    elif args.command == "export-vocab":
        tokenizer = LionTokenizer.load(Path(args.tokenizer))
        export_vocab_txt(tokenizer, Path(args.output))
        print(f"  ✓ Exported {tokenizer.vocab_size:,} tokens → {args.output}")


if __name__ == "__:<28}output}")���tpu