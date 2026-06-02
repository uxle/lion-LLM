"""
LionAI Interactive Chat  [Enhanced]
=====================================
New vs v1:
  • Hardware auto-detection on startup
  • Contrastive search mode for better quality
  • Beam search mode for factual/structured answers
  • /hardware command: show detected hardware + recommendations
  • /quant command: switch quantization live
  • /mode command: switch generation mode (sample/contrastive/beam)
  • System prompt templates
  • Token/second display after each response
  • Memory pressure warnings
  • Smarter context window with recency bias
  • Conversation export in multiple formats (JSON, Markdown, TXT)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from model import LionLLM, ModelConfig, InferenceEngine
from tokenizer import LionTokenizer
from memory import MemoryManager
from knowledge import KnowledgeEngine
from config import SystemConfig, detect_hardware, setup_logging, MemoryMonitor

logger = logging.getLogger(__name__)

# ── ANSI colors ──────────────────────────────────────────────────────────────
def _color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_C = {k: (v if _color() else "") for k, v in {
    "reset":   "\033[0m",  "bold":    "\033[1m",  "dim":     "\033[2m",
    "cyan":    "\033[96m", "green":   "\033[92m", "yellow":  "\033[93m",
    "red":     "\033[91m", "blue":    "\033[94m", "magenta": "\033[95m",
    "orange":  "\033[33m",
}.items()}

def _c(col: str, text: str) -> str:
    return f"{_C.get(col,'')}{text}{_C['reset']}"

BANNER = r"""
  _     _               _    ___
 | |   (_) ___  _ __   / \  |_ _|
 | |   | |/ _ \| '_ \ / _ \  | |
 | |___| | (_) | | | / ___ \ | |
 |_____|_|\___/|_| |_/_/   \_\___|
        Lion LLM  (LLLM)
"""

HELP = """
{bold}LionAI Commands{reset}

  {cyan}/help{reset}                  This help message
  {cyan}/reset{reset}                 Clear conversation history
  {cyan}/save [name]{reset}           Save session
  {cyan}/load [name]{reset}           Load session
  {cyan}/memory{reset}                List stored memories
  {cyan}/learn KEY VALUE{reset}       Store a memory
  {cyan}/forget KEY{reset}            Delete a memory
  {cyan}/stats{reset}                 Model + session statistics
  {cyan}/hardware{reset}              Hardware profile and recommendations
  {cyan}/config KEY=VAL{reset}        Tune generation (temp, top_k, top_p, min_p,
                           max_tokens, rep_pen, freq_pen)
  {cyan}/mode [sample|contrastive|beam]{reset}  Switch generation mode
  {cyan}/quant [none|fp16|int8|int4]{reset}     Change quantization (restarts engine)
  {cyan}/docs [path]{reset}           Ingest document(s) into knowledge base
  {cyan}/search QUERY{reset}          Search knowledge base
  {cyan}/export [file] [format]{reset} Export chat (format: json|md|txt)
  {cyan}/system PROMPT{reset}         Set system prompt
  {cyan}/exit{reset}                  Exit LionAI

{dim}Tip: /mode contrastive gives higher quality, less repetitive responses.
     /mode beam gives deterministic, factual answers (best for Q&A).{reset}
"""

SYSTEM_TEMPLATES = {
    "default":   "You are LionAI, a helpful and knowledgeable AI assistant running locally. Be concise, accurate, and helpful.",
    "assistant": "You are LionAI, a friendly personal assistant. Help the user efficiently and clearly.",
    "coder":     "You are LionAI, an expert software engineer. Provide correct, clean, well-commented code. Explain your choices briefly.",
    "research":  "You are LionAI, a research assistant. Provide thorough, well-reasoned analysis. Cite relevant concepts and note uncertainties.",
    "teacher":   "You are LionAI, a patient and clear teacher. Explain concepts step by step, use examples, and check understanding.",
}


# ─────────────────────────────────────────────
#  Generation Config
# ─────────────────────────────────────────────

class GenConfig:
    MODES = ("sample", "contrastive", "beam")

    def __init__(self) -> None:
        self.temperature       = 0.8
        self.top_k             = 40
        self.top_p             = 0.92
        self.min_p             = 0.05
        self.max_new_tokens    = 256
        self.repetition_penalty = 1.15
        self.frequency_penalty  = 0.1
        self.contrastive_alpha  = 0.6
        self.contrastive_k      = 5
        self.num_beams          = 4
        self.mode               = "sample"

    def update(self, key: str, value: str) -> str:
        aliases = {
            "temp": "temperature", "t": "temperature",
            "top_k": "top_k", "k": "top_k",
            "top_p": "top_p", "p": "top_p",
            "min_p": "min_p",
            "max": "max_new_tokens", "max_tokens": "max_new_tokens",
            "rep": "repetition_penalty", "rep_pen": "repetition_penalty",
            "freq": "frequency_penalty", "freq_pen": "frequency_penalty",
            "alpha": "contrastive_alpha", "beams": "num_beams",
        }
        attr = aliases.get(key.lower(), key.lower())
        if not hasattr(self, attr):
            return f"Unknown: {key}. Options: temp, top_k, top_p, min_p, max_tokens, rep_pen, freq_pen"
        try:
            setattr(self, attr, float(value) if "." in value else int(float(value))
                    if attr not in ("temperature","top_p","min_p","repetition_penalty",
                                    "frequency_penalty","contrastive_alpha") else float(value))
            return _c("green", f"✓ {attr} = {value}")
        except ValueError:
            return _c("red", f"Invalid value: {value}")

    def display(self) -> str:
        return (f"  mode              = {_c('cyan', self.mode)}\n"
                f"  temperature       = {self.temperature}\n"
                f"  top_k             = {self.top_k}\n"
                f"  top_p             = {self.top_p}\n"
                f"  min_p             = {self.min_p}\n"
                f"  max_new_tokens    = {self.max_new_tokens}\n"
                f"  repetition_penalty= {self.repetition_penalty}\n"
                f"  frequency_penalty = {self.frequency_penalty}\n"
                f"  contrastive_alpha = {self.contrastive_alpha}  (mode=contrastive)\n"
                f"  num_beams         = {self.num_beams}  (mode=beam)")


# ─────────────────────────────────────────────
#  Chat Session
# ─────────────────────────────────────────────

class ChatSession:
    def __init__(self, sessions_dir: Path, memory: MemoryManager) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.memory   = memory
        self._history: List[Dict] = []

    def add(self, role: str, content: str, meta: Optional[Dict] = None) -> None:
        self._history.append({"role": role, "content": content,
                               "ts": time.time(), **(meta or {})})

    def save(self, name: str) -> Path:
        path = self.sessions_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)
        self.memory.save_session(name)
        return path

    def load(self, name: str) -> bool:
        path = self.sessions_dir / f"{name}.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            self._history = json.load(f)
        self.memory.short.reset()
        for m in self._history[-30:]:
            self.memory.short.add(m["role"], m["content"])
        return True

    def export(self, path: Optional[Path] = None,
               fmt: str = "json") -> Path:
        path = path or (self.sessions_dir / f"export_{int(time.time())}.{fmt}")
        if fmt == "md":
            lines = ["# LionAI Conversation Export\n"]
            for m in self._history:
                role = m["role"].capitalize()
                lines.append(f"## {role}\n{m['content']}\n")
            path.write_text("\n".join(lines), encoding="utf-8")
        elif fmt == "txt":
            lines = [f"{m['role'].upper()}: {m['content']}" for m in self._history]
            path.write_text("\n\n".join(lines), encoding="utf-8")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        return path

    def list_sessions(self) -> List[str]:
        return [p.stem for p in sorted(self.sessions_dir.glob("*.json"))]

    def reset(self) -> None:
        self._history.clear()
        self.memory.short.reset()


# ─────────────────────────────────────────────
#  Chatbot
# ─────────────────────────────────────────────

class Chatbot:
    def __init__(self, model_dir: Path, data_dir: Path = Path("./data"),
                 device: Optional[str] = None,
                 quantize: str = "none",
                 config: Optional[SystemConfig] = None) -> None:
        model_dir = Path(model_dir)
        data_dir  = Path(data_dir)

        self._sys_config = config or SystemConfig.from_hardware()

        # ── Load tokenizer ───────────────────
        self.tokenizer = LionTokenizer.load(model_dir)

        # ── Load model via smart loader ──────
        from optimization import load_model_efficient
        quant  = quantize if quantize != "none" else self._sys_config.quantization
        dev    = device or self._sys_config.device
        gpu_l  = (self._sys_config.gpu_layers
                  if self._sys_config.gpu_layers >= 0 else None)

        print(_c("dim", f"  Loading model (quant={quant}, device={dev}) …"))
        model = load_model_efficient(model_dir, quantization=quant,
                                     device=dev, gpu_layers=gpu_l)

        self.engine = InferenceEngine(model, device=dev if quant != "int8" else "cpu")
        self._device = self.engine.device
        self._model_dir = model_dir
        self._quant = quant

        # ── Sub-systems ──────────────────────
        self.memory    = MemoryManager(data_dir / "memory",
                                       system_prompt=self._sys_config.system_prompt)
        self.knowledge = KnowledgeEngine(data_dir / "knowledge")
        self.session   = ChatSession(data_dir / "sessions", self.memory)
        self.gen_cfg   = GenConfig()
        self.monitor   = MemoryMonitor()
        self._running  = True
        self._resp_times: List[float] = []

    # ─── Generation ─────────────────────────
    def _generate(self, prompt: str) -> str:
        ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        stop_ids  = [self.tokenizer.EOS_ID,
                     self.tokenizer.token2id.get("</ast>", -1),
                     self.tokenizer.token2id.get("</s>", -1)]
        stop_ids  = [s for s in stop_ids if s >= 0]

        print(_c("green", "LionAI: "), end="", flush=True)
        t0 = time.perf_counter()
        out_ids: List[int] = []

        cfg = self.gen_cfg

        if cfg.mode == "beam":
            out_ids = self.engine.generate_beam(
                input_ids, max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
            )
            decoded = self.tokenizer.decode(out_ids)
            print(decoded, end="", flush=True)
        else:
            for tok_id in self.engine.generate(
                input_ids,
                max_new_tokens    = cfg.max_new_tokens,
                temperature       = cfg.temperature,
                top_k             = cfg.top_k,
                top_p             = cfg.top_p,
                min_p             = cfg.min_p,
                repetition_penalty= cfg.repetition_penalty,
                frequency_penalty = cfg.frequency_penalty,
                stop_ids          = stop_ids,
                contrastive_alpha = cfg.contrastive_alpha if cfg.mode == "contrastive" else 0.0,
                contrastive_k     = cfg.contrastive_k,
            ):
                out_ids.append(tok_id)
                print(self.tokenizer.decode([tok_id]), end="", flush=True)

        elapsed = time.perf_counter() - t0
        tps     = len(out_ids) / max(elapsed, 1e-6)
        self._resp_times.append(elapsed)

        print(f"\n{_c('dim', f'  [{len(out_ids)} tok | {tps:.0f} tok/s | {elapsed:.1f}s]')}")

        # Memory pressure warning
        if self.monitor.is_warn():
            info = self.monitor.check_ram()
            print(_c("yellow", f"  ⚠ Memory pressure high — consider /quant int8"))

        return self.tokenizer.decode(out_ids)

    def _build_prompt(self, user_input: str) -> str:
        rag_ctx = self.knowledge.format_context(user_input, top_k=2) if self._sys_config.enable_rag else ""
        mem_ctx = self.memory.recall(user_input, top_k=2)

        sys = self.memory.short.system_prompt
        if rag_ctx: sys += f"\n\n{rag_ctx}"
        if mem_ctx: sys += f"\n\n{mem_ctx}"
        self.memory.short.system_prompt = sys

        return self.memory.short.get_prompt()

    # ─── Commands ───────────────────────────
    def _handle_command(self, line: str) -> bool:
        parts = line.strip().split(None, 2)
        cmd   = parts[0].lower()

        if cmd == "/help":
            print(HELP.format(**_C))

        elif cmd == "/reset":
            self.session.reset()
            print(_c("yellow", "  [Conversation reset]"))

        elif cmd == "/save":
            name = parts[1] if len(parts) > 1 else f"session_{int(time.time())}"
            p = self.session.save(name)
            print(_c("green", f"  Saved → {p}"))

        elif cmd == "/load":
            if len(parts) < 2:
                ss = self.session.list_sessions()
                print("  Sessions:", ", ".join(ss) if ss else "(none)")
            else:
                ok = self.session.load(parts[1])
                print(_c("green" if ok else "red",
                         f"  {'Loaded' if ok else 'Not found'}: {parts[1]}"))

        elif cmd == "/memory":
            mems = self.memory.long.list_all()
            if not mems:
                print("  (no stored memories)")
            else:
                for m in mems:
                    print(f"  {_c('cyan', m.key)}: {m.value}  {_c('dim', m.category)}")

        elif cmd == "/learn":
            if len(parts) < 3:
                print("  Usage: /learn KEY VALUE")
            else:
                self.memory.remember(parts[1], parts[2])
                print(_c("green", f"  Stored: {parts[1]} = {parts[2]}"))

        elif cmd == "/forget":
            if len(parts) < 2: print("  Usage: /forget KEY")
            else:
                ok = self.memory.long.delete(parts[1])
                print(_c("green" if ok else "red",
                         f"  {'Deleted' if ok else 'Not found'}: {parts[1]}"))

        elif cmd == "/stats":
            mem  = self.memory.full_stats()
            know = self.knowledge.stats()
            hw   = self.monitor.check_ram()
            print(_c("bold", "\n  ── LionAI Statistics ──"))
            print(f"  Model:         {self._model_dir.name}  ({self._quant})")
            print(f"  Device:        {self._device}")
            print(f"  Mode:          {self.gen_cfg.mode}")
            if self._resp_times:
                avg_tps = (self.gen_cfg.max_new_tokens /
                           max(sum(self._resp_times) / len(self._resp_times), 1e-6))
                print(f"  Avg speed:     ~{avg_tps:.0f} tok/s")
            print(f"  Session turns: {mem['short_term']['turns']}")
            print(f"  Memories:      {mem['long_term']['total_memories']}")
            print(f"  Knowledge:     {know.get('chunks',0)} chunks / {know.get('documents',0)} docs")
            if "ram_used_gb" in hw:
                print(f"  RAM:           {hw['ram_used_gb']:.1f}/{hw['ram_total_gb']:.1f} GB")
            if "vram_used_gb" in hw:
                print(f"  VRAM:          {hw['vram_used_gb']:.1f}/{hw['vram_total_gb']:.1f} GB")
            print()

        elif cmd == "/hardware":
            hw = detect_hardware()
            print(_c("bold", "\n  ── Hardware Profile ──"))
            print(f"  CPU cores:     {hw.cpu_cores}")
            print(f"  RAM:           {hw.ram_gb:.1f} GB")
            print(f"  VRAM:          {hw.vram_gb:.1f} GB")
            print(f"  Device:        {hw.device}")
            print(_c("bold", "\n  ── Recommendations ──"))
            print(f"  Model size:    {_c('cyan', hw.recommended_model_size)}")
            print(f"  Quantization:  {_c('cyan', hw.recommended_quantization)}")
            print(f"  Batch size:    {hw.recommended_batch_size}")
            print(f"  Seq length:    {hw.recommended_seq_len}")
            print()

        elif cmd == "/config":
            if len(parts) < 2:
                print(_c("bold", "  Current Config:"))
                print(self.gen_cfg.display())
            else:
                for item in " ".join(parts[1:]).split():
                    if "=" in item:
                        k, v = item.split("=", 1)
                        from config import InputValidator
                        try:
                            v_clean = InputValidator().validate_config_value(k, v)
                            print("  " + self.gen_cfg.update(k, str(v_clean)))
                        except ValueError as e:
                            print(_c("red", f"  Error: {e}"))

        elif cmd == "/mode":
            if len(parts) < 2:
                print(f"  Current mode: {_c('cyan', self.gen_cfg.mode)}")
                print(f"  Available: {', '.join(GenConfig.MODES)}")
            else:
                m = parts[1].lower()
                if m in GenConfig.MODES:
                    self.gen_cfg.mode = m
                    print(_c("green", f"  Mode → {m}"))
                else:
                    print(_c("red", f"  Unknown mode: {m}"))

        elif cmd == "/quant":
            if len(parts) < 2:
                print(f"  Current: {self._quant}")
            else:
                q = parts[1].lower()
                print(_c("yellow", f"  Reloading model with quantization={q} …"))
                from optimization import load_model_efficient
                try:
                    new_model = load_model_efficient(
                        self._model_dir, quantization=q,
                        device=self._device if q != "int8" else "cpu"
                    )
                    self.engine = InferenceEngine(new_model)
                    self._quant = q
                    print(_c("green", f"  Quantization → {q}"))
                except Exception as e:
                    print(_c("red", f"  Failed: {e}"))

        elif cmd == "/system":
            if len(parts) < 2:
                print(f"  Templates: {', '.join(SYSTEM_TEMPLATES)}")
                print(f"  Current: {self.memory.short.system_prompt[:80]}…")
            else:
                arg = parts[1]
                if arg in SYSTEM_TEMPLATES:
                    self.memory.short.system_prompt = SYSTEM_TEMPLATES[arg]
                    print(_c("green", f"  System prompt → {arg} template"))
                else:
                    # Use raw text
                    self.memory.short.system_prompt = " ".join(parts[1:])
                    print(_c("green", "  System prompt updated"))

        elif cmd == "/docs":
            if len(parts) < 2:
                docs = self.knowledge.list_documents()
                if docs:
                    print(_c("bold", "  Indexed Documents:"))
                    for d in docs:
                        print(f"    {_c('cyan', Path(d['path']).name)}  {d['chunk_count']} chunks")
                else:
                    print("  (no documents indexed)")
            else:
                p = Path(parts[1])
                try:
                    if p.is_dir():
                        r = self.knowledge.ingest_directory(p)
                        for n, k in r.items():
                            print(f"  {n}: {k} chunks")
                    else:
                        n = self.knowledge.ingest_file(p)
                        print(_c("green", f"  Indexed {n} chunks from {p.name}"))
                except Exception as e:
                    print(_c("red", f"  Error: {e}"))

        elif cmd == "/search":
            if len(parts) < 2:
                print("  Usage: /search QUERY")
            else:
                q = " ".join(parts[1:])
                for r in self.knowledge.retrieve(q, top_k=4):
                    src = Path(r["source"]).name if r["source"] else "?"
                    print(f"\n  [{_c('cyan', src)}]\n  {r['text'][:200]} …")

        elif cmd == "/export":
            fmt  = "json"
            path = None
            if len(parts) >= 3: fmt  = parts[2]
            if len(parts) >= 2 and not parts[1].startswith("json|md|txt"):
                path = Path(parts[1])
            p = self.session.export(path, fmt)
            print(_c("green", f"  Exported → {p}"))

        elif cmd in ("/exit", "/quit", "/q", "/bye"):
            self.memory.save_session()
            print(_c("dim", "  Goodbye!"))
            self._running = False

        else:
            print(_c("yellow", f"  Unknown command: {cmd}  (type /help)"))

        return True

    # ─── Main loop ──────────────────────────
    def run(self) -> None:
        print(_c("cyan", BANNER))
        hw = detect_hardware()
        print(_c("bold",
                 f"  LionAI — Lion LLM  |  {self._device.upper()}  |  quant={self._quant}"))
        print(_c("dim",
                 f"  mode={self.gen_cfg.mode}  |  RAM={hw.ram_gb:.0f}GB  |  type /help\n"))

        while self._running:
            try:
                user_input = input(_c("blue", "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.memory.save_session()
                break

            if not user_input:
                continue
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Validate input
            from config import InputValidator
            try:
                user_input = InputValidator(8192).validate_text(user_input)
            except ValueError as e:
                print(_c("red", f"  Input rejected: {e}"))
                continue

            self.memory.short.add("user", user_input)
            self._sys_config.enable_rag = self.knowledge.stats().get("documents", 0) > 0
            prompt = self._build_prompt(user_input)

            try:
                response = self._generate(prompt)
            except Exception as e:
                logger.error("Generation error: %s", e, exc_info=True)
                print(_c("red", f"  [Error: {e}]"))
                continue

            self.memory.short.add("assistant", response)
            self.session.add("user",      user_input)
            self.session.add("assistant", response)


# ─────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LionAI — Lion LLM Chat",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",    default="./runs/lionai/final")
    parser.add_argument("--data",     default="./data")
    parser.add_argument("--device",   default=None, choices=["cpu","cuda","mps","auto"])
    parser.add_argument("--quantize", default="auto",
                        choices=["none","auto","fp16","bf16","int8","int4"])
    parser.add_argument("--mode",     default="sample",
                        choices=["sample","contrastive","beam"])
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--config",   default=None, help="Path to config.json")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "WARNING",
                  log_to_file=False)

    model_path = Path(args.model)
    if not model_path.exists():
        print(_c("red", f"  Model not found: {model_path}"))
        print(_c("dim", "  Run: python demo_setup.py  to create a demo model"))
        sys.exit(1)

    cfg = (SystemConfig.load(Path(args.config))
           if args.config else SystemConfig.from_hardware())

    quant = args.quantize
    if quant == "auto":
        quant = cfg.quantization

    bot = Chatbot(model_dir=model_path, data_dir=Path(args.data),
                  device=args.device, quantize=quant, config=cfg)
    bot.gen_cfg.mode = args.mode
    bot.run()


if __name__ == "__main__":
    main()
