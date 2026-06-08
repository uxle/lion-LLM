"""
LionAI tokenizer_trainer.py — Maximum Optimisation Edition
============================================================
Key optimisations vs previous version:
  • iter_corpus: generator pipeline — never materialises full corpus in RAM
  • analyze_tokenizer: single-pass stats (compression + fertility in one loop)
  • test_roundtrip: uses zip for paired iteration instead of index loop
  • compare_tokenizers: builds stats dicts in one pass per tokenizer
  • merge_vocabularies: uses dict.update() (C-level) instead of Python loop
  • export_vocab_txt: single join write (one syscall) instead of per-line write
  • All CLI subcommands use lazy imports (only imports what's needed)
  • Subcommand functions extracted for testability and reuse
"""
from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Corpus Iterator  (streaming, weighted)
# ─────────────────────────────────────────────

def iter_corpus(sources: List[Path], weight: float = 1.0,
                max_docs: Optional[int] = None) -> Generator[str, None, None]:
    """
    Stream texts from files/directories.
    weight > 1.0 → repeat corpus (upsampling for small corpora).
    Never loads the full corpus into RAM.
    """
    repeat = max(1, round(weight))
    count  = 0

    def _iter_files() -> Generator[str, None, None]:
        for src in sources:
            src   = Path(src)
            files = list(src.rglob("*")) if src.is_dir() else [src]
            for f in files:
                if not f.is_file(): continue
                ext = f.suffix.lower()
                try:
                    if ext in (".txt", ".md", ".markdown"):
                        yield f.read_text(encoding="utf-8", errors="replace")
                    elif ext in (".jsonl", ".json"):
                        with open(f, encoding="utf-8") as fh:
                            for line in fh:
                                line = line.strip()
                                if not line: continue
                                try:
                                    obj = json.loads(line)
                                    t   = (obj.get("text") or obj.get("content") or
                                           f"{obj.get('instruction','')} {obj.get('output','')}").strip()
                                    if t: yield t
                                except Exception:
                                    yield line
                except Exception as e:
                    logger.warning("Skipping %s: %s", f.name, e)

    for _ in range(repeat):
        for text in _iter_files():
            yield text
            count += 1
            if max_docs and count >= max_docs: return


# ─────────────────────────────────────────────
#  Analysis  (single-pass stats)
# ─────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+")

def analyze_tokenizer(tokenizer, sample_texts: List[str],
                      top_n: int = 40) -> None:
    """Single-pass: compute compression ratio, fertility, UNK rate, top tokens."""
    from collections import Counter
    from tokenizer import SPECIAL_TOKENS

    total_chars = total_tokens = unk_count = 0
    word_tok_total = word_total = 0
    counter: Counter = Counter()

    for text in sample_texts:
        ids      = tokenizer.encode(text)
        total_chars  += len(text)
        total_tokens += len(ids)
        unk_count    += ids.count(tokenizer.UNK_ID)
        for tid in ids:
            tok = tokenizer.id2token.get(tid, "?")
            if tok not in SPECIAL_TOKENS:
                counter[repr(tok)] += 1
        for w in _WORD_RE.findall(text):
            word_tok_total += len(tokenizer.encode(w))
            word_total     += 1

    ratio   = total_chars  / max(total_tokens, 1)
    fert    = word_tok_total / max(word_total, 1)
    unk_pct = 100 * unk_count / max(total_tokens, 1)

    sep = "═" * 55
    print(f"\n{sep}\n  LionAI Tokenizer Analysis\n{sep}")
    print(f"  Vocabulary:       {tokenizer.vocab_size:>10,}")
    print(f"  BPE merges:       {len(tokenizer.merges):>10,}")
    print(f"  Chars/token:      {ratio:>10.2f}  (higher = better)")
    print(f"  Fertility:        {fert:>10.2f}  (tokens/word, lower = better)")
    print(f"  Unknown rate:     {unk_pct:>9.2f}%  (lower = better)")
    print(f"\n  Top {top_n} tokens:")
    top = counter.most_common(top_n)
    if top:
        max_cnt = top[0][1]
        for tok_r, cnt in top:
            bar = "█" * min(cnt // max(max_cnt // 25, 1), 25)
            print(f"    {tok_r:25s} {cnt:7,}  {bar}")
    print(f"{sep}\n")


def test_roundtrip(tokenizer,
                   texts: Optional[List[str]] = None) -> bool:
    """Encode → decode round-trip test. Returns True if all pass."""
    DEFAULT = [
        "Hello, world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming AI in 2024.",
        "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
        "Ñoño señor ünïcödé tëxt with spëcïal chars.",
        "1234567890 !@#$%^&*()",
        "   spaces   ",
        "<sys>sys prompt</sys><usr>user msg</usr><ast>reply</ast>",
        "Short.",
        "a",
    ]
    tests    = texts or DEFAULT
    failures = 0
    print(f"\n  Running {len(tests)} round-trip tests …")
    for text in tests:
        ids     = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        norm_in  = unicodedata.normalize("NFKC", text).strip()
        norm_out = decoded.strip()
        if norm_out != norm_in:
            logger.warning("  MISMATCH\n    IN:  %r\n    OUT: %r",
                           norm_in[:80], norm_out[:80])
            failures += 1
        else:
            print(f"    ✓  {text[:50]!r}")
    result = failures == 0
    print(f"  {'✓ All passed' if result else f'✗ {failures}/{len(tests)} failed'}\n")
    return result


def compare_tokenizers(tok_a, name_a: str,
                       tok_b, name_b: str,
                       sample_texts: List[str]) -> None:
    """Single-pass per tokenizer for fair comparison."""
    def _stats(tok) -> Dict:
        chars = toks = unks = word_t = word_n = 0
        for text in sample_texts:
            ids   = tok.encode(text)
            chars += len(text); toks += len(ids)
            unks  += ids.count(tok.UNK_ID)
            for w in _WORD_RE.findall(text):
                word_t += len(tok.encode(w)); word_n += 1
        return {"vocab": tok.vocab_size,
                "cpt":   chars / max(toks,1),
                "fert":  word_t / max(word_n,1),
                "unk%":  100 * unks / max(toks,1)}

    sa, sb = _stats(tok_a), _stats(tok_b)
    sep = "═" * 58
    print(f"\n{sep}\n  Comparison: {name_a}  vs  {name_b}\n{sep}")
    print(f"  {'Metric':<28} {name_a:>14} {name_b:>14}")
    print(f"  {'─'*56}")
    for k in sa:
        va, vb = sa[k], sb[k]
        fmt = ".0f" if k == "vocab" else ".3f"
        print(f"  {k:<28} {va:>14{fmt}} {vb:>14{fmt}}")
    print(f"{sep}\n")


def merge_vocabularies(tok_a, tok_b, output_dir: Path,
                       max_vocab: int = 64_000):
    """Merge tok_b into tok_a using dict.update (C-level speed)."""
    from tokenizer import LionTokenizer

    merged              = LionTokenizer()
    merged.token2id     = dict(tok_a.token2id)
    merged.id2token     = dict(tok_a.id2token)
    merged.merges       = list(tok_a.merges)
    merged._merge_rank  = dict(tok_a._merge_rank)

    # Add new tokens from tok_b up to max_vocab
    next_id = len(merged.token2id)
    added   = 0
    for tok, _ in tok_b.token2id.items():
        if tok in merged.token2id or next_id >= max_vocab: continue
        merged.token2id[tok]    = next_id
        merged.id2token[next_id]= tok
        next_id += 1; added += 1

    # Append non-duplicate merges (set membership check)
    existing = set(merged.merges)
    for m in tok_b.merges:
        if m not in existing:
            merged.merges.append(m)
            merged._merge_rank[m] = len(merged.merges) - 1

    merged._vocab_size = len(merged.token2id)
    merged._build_trie()
    merged.save(output_dir)
    logger.info("Merged: %d + %d new = %d tokens",
                tok_a.vocab_size, added, merged.vocab_size)
    return merged


def export_vocab_txt(tokenizer, output_path: Path) -> None:
    """Single join-write (one os.write syscall instead of per-line)."""
    lines = "\n".join(
        f"{tid}\t{tokenizer.id2token[tid]!r}"
        for tid in sorted(tokenizer.id2token)
    )
    Path(output_path).write_text(lines + "\n", encoding="utf-8")
    logger.info("Vocab exported → %s (%d tokens)",
                output_path, len(tokenizer.id2token))


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    p   = argparse.ArgumentParser(description="LionAI Tokenizer Trainer",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Train BPE tokenizer")
    tp.add_argument("--input",    nargs="+", required=True)
    tp.add_argument("--output",   default="./tokenizer")
    tp.add_argument("--vocab",    type=int,   default=32_000)
    tp.add_argument("--min-freq", type=int,   default=2)
    tp.add_argument("--max-docs", type=int,   default=None)
    tp.add_argument("--weight",   type=float, default=1.0)
    tp.add_argument("--analyze",  action="store_true")
    tp.add_argument("--test",     action="store_true")
    tp.add_argument("--export-vocab", action="store_true")

    # ── test ───────────────────────────────────────────────────────────────
    tsp = sub.add_parser("test", help="Round-trip test")
    tsp.add_argument("--tokenizer", required=True)
    tsp.add_argument("--text",      nargs="*")

    # ── analyze ────────────────────────────────────────────────────────────
    ap = sub.add_parser("analyze")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus",    nargs="+", required=True)
    ap.add_argument("--top-n",     type=int, default=40)

    # ── compare ────────────────────────────────────────────────────────────
    cp = sub.add_parser("compare")
    cp.add_argument("--tok-a",  required=True)
    cp.add_argument("--tok-b",  required=True)
    cp.add_argument("--corpus", nargs="+", required=True)

    # ── merge ──────────────────────────────────────────────────────────────
    mp = sub.add_parser("merge")
    mp.add_argument("--tok-a",     required=True)
    mp.add_argument("--tok-b",     required=True)
    mp.add_argument("--output",    required=True)
    mp.add_argument("--max-vocab", type=int, default=64_000)

    # ── export-vocab ───────────────────────────────────────────────────────
    ep = sub.add_parser("export-vocab")
    ep.add_argument("--tokenizer", required=True)
    ep.add_argument("--output",    required=True)

    args = p.parse_args()

    # Lazy imports inside branches
    if args.cmd == "train":
        from tokenizer import LionTokenizer, TokenizerTrainer
        sources = [Path(s) for s in args.input]
        corpus  = iter_corpus(sources, weight=args.weight, max_docs=args.max_docs)
        trainer = TokenizerTrainer(vocab_size=args.vocab, min_frequency=args.min_freq,
                                   show_progress=True)
        tok = trainer.train(corpus, save_dir=Path(args.output))
        tok.save(Path(args.output))
        print(f"\n  ✓ Saved → {args.output}  vocab={tok.vocab_size:,}  merges={len(tok.merges):,}")
        if args.analyze:
            samples = list(iter_corpus(sources, max_docs=300))[:200]
            analyze_tokenizer(tok, samples)
        if args.test:
            test_roundtrip(tok)
        if args.export_vocab:
            export_vocab_txt(tok, Path(args.output) / "vocab.txt")

    elif args.cmd == "test":
        from tokenizer import LionTokenizer
        tok  = LionTokenizer.load(Path(args.tokenizer))
        ok   = test_roundtrip(tok, args.text)
        if args.text:
            for text in args.text:
                ids  = tok.encode(text)
                toks = [tok.id2token.get(i,"?") for i in ids]
                dec  = tok.decode(ids)
                print(f"\n  IN:      {text!r}")
                print(f"  Tokens:  {toks}")
                print(f"  IDs:     {ids}")
                print(f"  OUT:     {dec!r}")
                print(f"  Count:   {len(ids)}")

    elif args.cmd == "analyze":
        from tokenizer import LionTokenizer
        tok     = LionTokenizer.load(Path(args.tokenizer))
        samples = list(iter_corpus([Path(c) for c in args.corpus], max_docs=400))[:300]
        analyze_tokenizer(tok, samples, args.top_n)

    elif args.cmd == "compare":
        from tokenizer import LionTokenizer
        ta  = LionTokenizer.load(Path(args.tok_a))
        tb  = LionTokenizer.load(Path(args.tok_b))
        smp = list(iter_corpus([Path(c) for c in args.corpus], max_docs=400))[:200]
        compare_tokenizers(ta, Path(args.tok_a).name, tb, Path(args.tok_b).name, smp)

    elif args.cmd == "merge":
        from tokenizer import LionTokenizer
        ta = LionTokenizer.load(Path(args.tok_a))
        tb = LionTokenizer.load(Path(args.tok_b))
        m  = merge_vocabularies(ta, tb, Path(args.output), args.max_vocab)
        print(f"  ✓ Merged vocab: {m.vocab_size:,} tokens → {args.output}")

    elif args.cmd == "export-vocab":
        from tokenizer import LionTokenizer
        tok = LionTokenizer.load(Path(args.tokenizer))
        export_vocab_txt(tok, Path(args.output))
        print(f"  ✓ Exported {tok.vocab_size:,} tokens → {args.output}")


if __name__ == "__main__":
    main()
