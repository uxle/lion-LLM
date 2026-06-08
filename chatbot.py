"""
LionAI chatbot.py — Intelligent Real-Time Learning Edition
===========================================================
Integrates:
  • ReasoningPipeline (chain-of-thought, intent, verification)
  • OnlineLearner (real-time LoRA updates from every turn)
  • LoRA injection at startup (trainable adapter on frozen base)
  • Explicit /good and /bad feedback commands
  • /correct OLD → NEW correction command
  • /learn_stats to see learning progress
  • Confidence-aware response display
  • Intent displayed per turn (so user understands AI's interpretation)
  • Auto-LoRA checkpoint saved every 10 turns
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import torch

from model import LionLLM, InferenceEngine
from tokenizer import LionTokenizer, SPECIAL_TOKENS, _SPECIAL_SET
from memory import MemoryManager
from knowledge import KnowledgeEngine
from config import SystemConfig, detect_hardware, setup_logging, MemoryMonitor, InputValidator
from learner import OnlineLearner
from reasoner import ReasoningPipeline

logger = logging.getLogger(__name__)

# ── ANSI codes ───────────────────────────────────────────────────────────────
def _mk_c() -> Dict[str, str]:
    if not (sys.stdout.isatty() and not os.environ.get("NO_COLOR")):
        return {k: "" for k in ("rst","bold","dim","cyan","green","yellow","red","blue","magenta","orange")}
    return {"rst":"\033[0m","bold":"\033[1m","dim":"\033[2m","cyan":"\033[96m",
            "green":"\033[92m","yellow":"\033[93m","red":"\033[91m",
            "blue":"\033[94m","magenta":"\033[95m","orange":"\033[33m"}

C = _mk_c()
def _c(k: str, t: str) -> str: return f"{C.get(k,'')}{t}{C['rst']}"

BANNER = r"""
  _     _               _    ___
 | |   (_) ___  _ __   / \  |_ _|
 | |   | |/ _ \| '_ \ / _ \  | |
 | |___| | (_) | | | / ___ \ | |
 |_____|_|\___/|_| |_/_/   \_\___|
        Lion LLM  (LLLM)  v3
"""

HELP = f"""
{C['bold']}LionAI Commands{C['rst']}

  {C['cyan']}/help{C['rst']}                     This message
  {C['cyan']}/reset{C['rst']}                    Clear conversation
  {C['cyan']}/save [name]{C['rst']}              Save session
  {C['cyan']}/load [name]{C['rst']}              Load session

  {C['bold']}Learning & Feedback:{C['rst']}
  {C['cyan']}/good{C['rst']}                     Mark last response as good  ✓
  {C['cyan']}/bad{C['rst']}                      Mark last response as bad   ✗
  {C['cyan']}/correct "OLD" "NEW"{C['rst']}      Correct a wrong answer (teaches the AI)
  {C['cyan']}/learn_stats{C['rst']}              Show real-time learning statistics
  {C['cyan']}/save_lora{C['rst']}                Save learned LoRA weights

  {C['bold']}Memory:{C['rst']}
  {C['cyan']}/memory{C['rst']}                   List stored memories
  {C['cyan']}/learn KEY VALUE{C['rst']}          Store a memory
  {C['cyan']}/forget KEY{C['rst']}               Delete a memory

  {C['bold']}Knowledge Base:{C['rst']}
  {C['cyan']}/docs [path]{C['rst']}              Ingest document(s)
  {C['cyan']}/search QUERY{C['rst']}             Search knowledge base

  {C['bold']}Settings:{C['rst']}
  {C['cyan']}/stats{C['rst']}                    System statistics
  {C['cyan']}/hardware{C['rst']}                 Hardware profile
  {C['cyan']}/config KEY=VAL{C['rst']}           Tune generation
  {C['cyan']}/mode [sample|contrastive|beam]{C['rst']}
  {C['cyan']}/quant [none|int8|int4]{C['rst']}   Change quantization
  {C['cyan']}/system PROMPT{C['rst']}            Set system prompt
  {C['cyan']}/export [file]{C['rst']}            Export conversation
  {C['cyan']}/exit{C['rst']}                     Exit

{C['dim']}Tip: /good and /bad after each response teaches LionAI in real time.
     /correct lets you show the AI what the right answer should be.{C['rst']}
"""

_SYSTEM_TEMPLATES = {
    "default":   "You are LionAI, a helpful AI assistant. Be clear, accurate, and concise.",
    "coder":     "You are LionAI, an expert software engineer. Write correct, clean, well-commented code. Always include a usage example.",
    "teacher":   "You are LionAI, a patient and clear teacher. Explain step by step with examples. Check understanding.",
    "research":  "You are LionAI, a thorough research assistant. Be detailed, cite reasoning, note uncertainties.",
    "assistant": "You are LionAI, a friendly personal assistant. Be efficient, warm, and helpful.",
}


# ─────────────────────────────────────────────
#  Generation Config  (__slots__)
# ─────────────────────────────────────────────

class GenConfig:
    __slots__ = ("temperature","top_k","top_p","min_p","max_new_tokens",
                 "repetition_penalty","frequency_penalty","contrastive_alpha",
                 "contrastive_k","num_beams","mode","use_reasoning","show_intent",
                 "show_confidence","auto_learn","verify_responses")
    MODES = frozenset(("sample","contrastive","beam"))

    def __init__(self) -> None:
        self.temperature        = 0.8
        self.top_k              = 40
        self.top_p              = 0.92
        self.min_p              = 0.05
        self.max_new_tokens     = 256
        self.repetition_penalty = 1.1
        self.frequency_penalty  = 0.0
        self.contrastive_alpha  = 0.6
        self.contrastive_k      = 4
        self.num_beams          = 4
        self.mode               = "sample"
        self.use_reasoning      = True    # chain-of-thought
        self.show_intent        = True    # display detected intent
        self.show_confidence    = False   # display confidence score
        self.auto_learn         = True    # learn from every turn automatically
        self.verify_responses   = True    # self-verify before showing

    _ALIASES = {
        "temp":"temperature","t":"temperature","k":"top_k","p":"top_p",
        "max":"max_new_tokens","max_tokens":"max_new_tokens",
        "rep":"repetition_penalty","rep_pen":"repetition_penalty",
        "freq":"frequency_penalty","alpha":"contrastive_alpha",
        "beams":"num_beams",
        "reasoning":"use_reasoning","learn":"auto_learn",
        "intent":"show_intent","verify":"verify_responses",
    }
    _FLOATS = frozenset(("temperature","top_p","min_p","repetition_penalty",
                          "frequency_penalty","contrastive_alpha"))
    _BOOLS  = frozenset(("use_reasoning","show_intent","show_confidence",
                          "auto_learn","verify_responses"))

    def update(self, key: str, value: str) -> str:
        attr = self._ALIASES.get(key.lower(), key.lower())
        if not hasattr(self, attr):
            return _c("red", f"Unknown: {key}")
        try:
            if attr in self._BOOLS:
                v = value.lower() in ("1","true","yes","on")
            elif attr in self._FLOATS:
                v = float(value)
            else:
                v = int(float(value))
            setattr(self, attr, v)
            return _c("green", f"✓ {attr} = {v}")
        except ValueError:
            return _c("red", f"Invalid: {value}")

    def display(self) -> str:
        lines = []
        for a in self.__slots__:
            v = getattr(self, a)
            lines.append(f"  {a:<28}= {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Chat Session
# ─────────────────────────────────────────────

class ChatSession:
    __slots__ = ("sessions_dir","memory","_history","_last_turn_id")

    def __init__(self, sessions_dir: Path, memory: MemoryManager) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.memory       = memory
        self._history:    List[Dict] = []
        self._last_turn_id: Optional[int] = None

    def add(self, role: str, content: str,
            meta: Optional[Dict] = None) -> None:
        self._history.append({
            "role": role, "content": content,
            "ts": time.time(), **(meta or {})
        })

    def set_last_turn_id(self, tid: int) -> None:
        self._last_turn_id = tid

    @property
    def last_turn_id(self) -> Optional[int]:
        return self._last_turn_id

    def save(self, name: str) -> Path:
        path = self.sessions_dir / f"{name}.json"
        path.write_text(json.dumps(self._history, indent=2), encoding="utf-8")
        self.memory.save_session(name)
        return path

    def load(self, name: str) -> bool:
        path = self.sessions_dir / f"{name}.json"
        if not path.exists(): return False
        self._history = json.loads(path.read_text(encoding="utf-8"))
        self.memory.short.reset()
        for m in self._history[-30:]:
            self.memory.short.add(m["role"], m["content"])
        return True

    def export(self, path: Optional[Path] = None, fmt: str = "json") -> Path:
        path = path or (self.sessions_dir / f"export_{int(time.time())}.{fmt}")
        if fmt == "md":
            path.write_text("\n\n".join(
                f"**{m['role'].capitalize()}:** {m['content']}"
                for m in self._history), encoding="utf-8")
        elif fmt == "txt":
            path.write_text("\n\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in self._history), encoding="utf-8")
        else:
            path.write_text(json.dumps(self._history, indent=2), encoding="utf-8")
        return path

    def list_sessions(self) -> List[str]:
        return [p.stem for p in sorted(self.sessions_dir.glob("*.json"))]

    def reset(self) -> None:
        self._history.clear(); self.memory.short.reset()
        self._last_turn_id = None

    @property
    def last_user(self) -> str:
        for m in reversed(self._history):
            if m["role"] == "user": return m["content"]
        return ""

    @property
    def last_assistant(self) -> str:
        for m in reversed(self._history):
            if m["role"] == "assistant": return m["content"]
        return ""


# ─────────────────────────────────────────────
#  Chatbot
# ─────────────────────────────────────────────

class Chatbot:
    def __init__(self, model_dir: Path,
                 data_dir:   Path = Path("./data"),
                 device:     Optional[str] = None,
                 quantize:   str  = "none",
                 config:     Optional[SystemConfig] = None,
                 lora_r:     int  = 8,
                 enable_learning: bool = True) -> None:

        model_dir = Path(model_dir); data_dir = Path(data_dir)

        self._hw         = detect_hardware()
        self._sys_cfg    = config or SystemConfig.from_hardware()
        self._validator  = InputValidator(8192)
        self._model_dir  = model_dir
        self._running    = True
        self._turn_count = 0

        # ── Load model ───────────────────────────────────────────────────────
        from optimization import load_model_efficient, inject_lora
        quant = quantize if quantize != "none" else self._sys_cfg.quantization
        dev   = device or self._sys_cfg.device
        gpu_l = self._sys_cfg.gpu_layers if self._sys_cfg.gpu_layers >= 0 else None

        print(_c("dim", f"\n  Loading LionAI (quant={quant} device={dev}) …"))
        self.tokenizer = LionTokenizer.load(model_dir)
        model          = load_model_efficient(model_dir, quant, dev, gpu_l)

        # ── Inject LoRA adapters for online learning ─────────────────────────
        if enable_learning and lora_r > 0:
            print(_c("dim", f"  Injecting LoRA adapters (r={lora_r}) for real-time learning …"))
            model = inject_lora(model, r=lora_r, alpha=float(lora_r * 2),
                                targets=["q_proj", "v_proj", "o_proj", "gate_up"])

        self.engine    = InferenceEngine(model, device=dev if quant != "int8" else "cpu")
        self._device   = self.engine.device
        self._quant    = quant

        # ── Sub-systems ──────────────────────────────────────────────────────
        self.memory    = MemoryManager(data_dir / "memory",
                                       system_prompt=self._sys_cfg.system_prompt)
        self.knowledge = KnowledgeEngine(data_dir / "knowledge")
        self.session   = ChatSession(data_dir / "sessions", self.memory)
        self.gen_cfg   = GenConfig()
        self.monitor   = MemoryMonitor()

        # ── Intelligence modules ─────────────────────────────────────────────
        self.reasoning = ReasoningPipeline(self.knowledge)

        # Online learner (only if LoRA was injected)
        if enable_learning and lora_r > 0:
            self.learner = OnlineLearner(
                model       = model,
                tokenizer   = self.tokenizer,
                data_dir    = data_dir,
                learning_rate = 5e-5,
                update_every  = 4,
                replay_every  = 20,
                max_seq       = 128,
            )
            # Try loading existing LoRA checkpoint
            lora_ckpt = model_dir / "lora_online"
            if lora_ckpt.exists():
                self.learner.load_checkpoint(lora_ckpt)
                print(_c("green", f"  ✓ LoRA checkpoint loaded ({self.learner._update_count} updates)"))
        else:
            self.learner = None

        # Pre-build stop ids
        self._stop_ids: FrozenSet[int] = frozenset(
            i for i in (self.tokenizer.EOS_ID,
                        self.tokenizer.token2id.get("</ast>", -1),
                        self.tokenizer.token2id.get("</s>", -1))
            if i >= 0
        )

        # Command dispatch table
        self._cmds = {
            "/help":        self._cmd_help,
            "/reset":       self._cmd_reset,
            "/save":        self._cmd_save,
            "/load":        self._cmd_load,
            "/memory":      self._cmd_memory,
            "/learn":       self._cmd_learn,
            "/forget":      self._cmd_forget,
            "/stats":       self._cmd_stats,
            "/hardware":    self._cmd_hardware,
            "/config":      self._cmd_config,
            "/mode":        self._cmd_mode,
            "/quant":       self._cmd_quant,
            "/docs":        self._cmd_docs,
            "/search":      self._cmd_search,
            "/export":      self._cmd_export,
            "/system":      self._cmd_system,
            "/exit":        self._cmd_exit,
            "/quit":        self._cmd_exit,
            # Learning commands
            "/good":        self._cmd_good,
            "/bad":         self._cmd_bad,
            "/correct":     self._cmd_correct,
            "/learn_stats": self._cmd_learn_stats,
            "/save_lora":   self._cmd_save_lora,
        }

    # ─── Generation ─────────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        ids       = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([ids], dtype=torch.long)
        sdec      = self.tokenizer.make_streaming_decoder()
        cfg       = self.gen_cfg

        print(_c("green", "LionAI: "), end="", flush=True)
        t0 = time.perf_counter()
        out_ids: List[int] = []

        if cfg.mode == "beam":
            out_ids = self.engine.generate_beam(
                input_ids, max_new_tokens=int(cfg.max_new_tokens),
                num_beams=cfg.num_beams)
            decoded = self.tokenizer.decode(out_ids)
            print(decoded, flush=True)
        else:
            for tok_id in self.engine.generate(
                input_ids,
                max_new_tokens     = int(cfg.max_new_tokens),
                temperature        = cfg.temperature,
                top_k              = int(cfg.top_k),
                top_p              = cfg.top_p,
                min_p              = cfg.min_p,
                repetition_penalty = cfg.repetition_penalty,
                frequency_penalty  = cfg.frequency_penalty,
                stop_ids           = list(self._stop_ids),
                contrastive_alpha  = (cfg.contrastive_alpha
                                      if cfg.mode == "contrastive" else 0.0),
                contrastive_k      = cfg.contrastive_k,
            ):
                out_ids.append(tok_id)
                chunk = sdec.push(self.tokenizer.id2token.get(tok_id, ""))
                if chunk: print(chunk, end="", flush=True)

        rem = sdec.flush()
        if rem: print(rem, end="", flush=True)

        elapsed = time.perf_counter() - t0
        tps     = len(out_ids) / max(elapsed, 1e-6)
        print(f"\n{_c('dim', f'  [{len(out_ids)} tok | {tps:.0f} tok/s | {elapsed:.1f}s]')}")

        if self.monitor.is_warn():
            print(_c("yellow", "  ⚠ Memory pressure — try /quant int8"))

        return self.tokenizer.decode(out_ids)

    def _process_turn(self, user_input: str) -> str:
        """
        Full turn pipeline:
          1. Intent classification + CoT reasoning
          2. Build augmented prompt
          3. Generate response
          4. Self-verify
          5. Online learn from this turn
        """
        cfg = self.gen_cfg

        # ── Step 1: Reasoning pipeline ──────────────────────────────────────
        has_docs   = self.knowledge.stats().get("documents", 0) > 0
        rag_ctx    = (self.knowledge.format_context(user_input, top_k=2)
                      if has_docs else "")
        mem_ctx    = (self.memory.recall(user_input, top_k=2)
                      if self.memory.short.turn_count > 0 else "")

        if cfg.use_reasoning and self.reasoning.should_use_cot(user_input):
            trace, augmented = self.reasoning.prepare(
                user_input, memory_context=mem_ctx, rag_context=rag_ctx
            )
            if cfg.show_intent:
                print(_c("dim",
                         f"  [intent: {trace.intent} | "
                         f"steps: {len(trace.steps)} | "
                         f"{trace.elapsed_ms:.0f}ms]"))
        else:
            from reasoner import ReasoningTrace
            trace = ReasoningTrace(query=user_input, intent="unknown")
            augmented = "\n\n".join(p for p in (rag_ctx, mem_ctx) if p)

        # ── Step 2: Build system prompt with reasoning context ───────────────
        sys_prompt = self.memory.short.system_prompt
        if augmented:
            sys_prompt = sys_prompt + "\n\n" + augmented
        self.memory.short.system_prompt = sys_prompt
        prompt = self.memory.short.get_prompt()

        # ── Step 3: Generate ─────────────────────────────────────────────────
        raw_response = self._generate(prompt)

        # ── Step 4: Self-verify ──────────────────────────────────────────────
        if cfg.verify_responses:
            verify_result, confidence, final_response = self.reasoning.evaluate(
                user_input, raw_response, trace
            )
            if cfg.show_confidence:
                q = verify_result["quality"]
                col = "green" if q >= 0.7 else ("yellow" if q >= 0.45 else "red")
                print(_c(col, f"  [quality: {q:.0%} | confidence: {confidence:.0%}]"))
            if verify_result["issues"]:
                for issue in verify_result["issues"]:
                    logger.debug("Verify: %s", issue)
        else:
            final_response = raw_response
            verify_result  = {}
            confidence     = 0.7

        # ── Step 5: Online learning ──────────────────────────────────────────
        if cfg.auto_learn and self.learner:
            reward_info = self.learner.observe(
                prompt=user_input,
                response=final_response,
                user_signal=0.5,  # neutral until user gives explicit feedback
            )
            # Store turn_id for /good and /bad commands
            self.session.set_last_turn_id(reward_info.get("stored_id"))

            # Auto-checkpoint every 10 turns
            if self._turn_count % 10 == 0 and self._turn_count > 0:
                self.learner.save_checkpoint(self._model_dir / "lora_online")

        self._turn_count += 1
        return final_response

    # ─── Command Handlers ───────────────────────────────────────────────────

    def _cmd_help(self, p: List[str]) -> None:
        print(HELP)

    def _cmd_reset(self, p: List[str]) -> None:
        self.session.reset()
        print(_c("yellow", "  [Conversation reset]"))

    def _cmd_save(self, p: List[str]) -> None:
        name = p[1] if len(p) > 1 else f"s_{int(time.time())}"
        print(_c("green", f"  Saved → {self.session.save(name)}"))

    def _cmd_load(self, p: List[str]) -> None:
        if len(p) < 2:
            print("  Sessions:", ", ".join(self.session.list_sessions()) or "(none)")
            return
        ok = self.session.load(p[1])
        print(_c("green" if ok else "red",
                 f"  {'Loaded' if ok else 'Not found'}: {p[1]}"))

    def _cmd_memory(self, p: List[str]) -> None:
        mems = self.memory.long.list_all()
        if not mems: print("  (no memories)"); return
        for m in mems:
            print(f"  {_c('cyan', m.key)}: {m.value}  {_c('dim', m.category)}")

    def _cmd_learn(self, p: List[str]) -> None:
        if len(p) < 3: print("  Usage: /learn KEY VALUE"); return
        self.memory.remember(p[1], " ".join(p[2:]))
        print(_c("green", f"  Stored: {p[1]}"))

    def _cmd_forget(self, p: List[str]) -> None:
        if len(p) < 2: print("  Usage: /forget KEY"); return
        ok = self.memory.long.delete(p[1])
        print(_c("green" if ok else "red",
                 f"  {'Deleted' if ok else 'Not found'}: {p[1]}"))

    def _cmd_stats(self, p: List[str]) -> None:
        mem  = self.memory.full_stats()
        know = self.knowledge.stats()
        hw   = self.monitor.check_ram()
        print(_c("bold", "\n  ── LionAI Stats ──"))
        print(f"  Device: {self._device}  quant={self._quant}  mode={self.gen_cfg.mode}")
        print(f"  Turns:  {self._turn_count}  |  Memories: {mem['long_term']['total_memories']}")
        print(f"  Knowledge: {know.get('chunks',0)} chunks / {know.get('documents',0)} docs")
        if "ram_used_gb" in hw:
            print(f"  RAM:    {hw['ram_used_gb']:.1f}/{hw['ram_total_gb']:.1f} GB")
        if "vram_used_gb" in hw:
            print(f"  VRAM:   {hw['vram_used_gb']:.1f}/{hw['vram_total_gb']:.1f} GB")
        if self.learner:
            ls = self.learner.stats()
            print(f"  Learning: {ls['update_count']} updates | avg_loss={ls['avg_train_loss']}")

    def _cmd_hardware(self, p: List[str]) -> None:
        hw = self._hw
        print(f"\n  CPU×{hw.cpu_cores}  RAM={hw.ram_gb:.1f}GB  VRAM={hw.vram_gb:.1f}GB")
        print(f"  Device={hw.device}{'  [AMD/ROCm]' if hw.is_amd else ''}")
        print(f"  Recommended: {hw.recommended_model_size}  quant={hw.recommended_quantization}\n")

    def _cmd_config(self, p: List[str]) -> None:
        if len(p) < 2:
            print(_c("bold", "  Config:")); print(self.gen_cfg.display()); return
        for item in " ".join(p[1:]).split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    v = str(self._validator.validate_config_value(k, v))
                except ValueError as e:
                    print(_c("red", f"  {e}")); continue
                print("  " + self.gen_cfg.update(k, v))

    def _cmd_mode(self, p: List[str]) -> None:
        if len(p) < 2:
            print(f"  mode={self.gen_cfg.mode}  options: {GenConfig.MODES}"); return
        m = p[1].lower()
        if m in GenConfig.MODES: self.gen_cfg.mode = m; print(_c("green", f"  Mode → {m}"))
        else: print(_c("red", f"  Unknown: {m}"))

    def _cmd_quant(self, p: List[str]) -> None:
        if len(p) < 2: print(f"  quant={self._quant}"); return
        q = p[1].lower()
        print(_c("yellow", f"  Reloading with quant={q} …"))
        from optimization import load_model_efficient
        try:
            m = load_model_efficient(self._model_dir, q,
                                      self._device if q != "int8" else "cpu")
            self.engine = InferenceEngine(m)
            self._quant = q
            print(_c("green", f"  Quant → {q}"))
        except Exception as e:
            print(_c("red", f"  Failed: {e}"))

    def _cmd_docs(self, p: List[str]) -> None:
        if len(p) < 2:
            docs = self.knowledge.list_documents()
            for d in docs:
                print(f"  {_c('cyan', Path(d['path']).name)}  {d['n_chunks']} chunks")
            if not docs: print("  (no documents)")
            return
        path = Path(p[1])
        try:
            if path.is_dir():
                r = self.knowledge.ingest_directory(path)
                for name, k in r.items(): print(f"  {name}: {k} chunks")
            else:
                n = self.knowledge.ingest_file(path)
                print(_c("green", f"  {n} chunks ← {path.name}"))
        except Exception as e:
            print(_c("red", f"  Error: {e}"))

    def _cmd_search(self, p: List[str]) -> None:
        if len(p) < 2: print("  Usage: /search QUERY"); return
        for r in self.knowledge.retrieve(" ".join(p[1:]), top_k=4):
            src = Path(r.get("source", "?")).name
            print(f"\n  [{_c('cyan', src)}]\n  {r['text'][:200]} …")

    def _cmd_export(self, p: List[str]) -> None:
        fmt  = p[2] if len(p) > 2 else "json"
        path = Path(p[1]) if len(p) > 1 else None
        print(_c("green", f"  Exported → {self.session.export(path, fmt)}"))

    def _cmd_system(self, p: List[str]) -> None:
        if len(p) < 2:
            print(f"  Templates: {', '.join(_SYSTEM_TEMPLATES)}")
            print(f"  Current: {self.memory.short.system_prompt[:80]}…"); return
        arg = p[1]
        if arg in _SYSTEM_TEMPLATES:
            self.memory.short.system_prompt = _SYSTEM_TEMPLATES[arg]
            print(_c("green", f"  → {arg} template"))
        else:
            self.memory.short.system_prompt = " ".join(p[1:])
            print(_c("green", "  System prompt updated"))

    def _cmd_exit(self, p: List[str]) -> None:
        self.memory.save_session()
        if self.learner:
            self.learner.save_checkpoint(self._model_dir / "lora_online")
        print(_c("dim", "  Goodbye!"))
        self._running = False

    # ── Learning commands ────────────────────────────────────────────────────

    def _cmd_good(self, p: List[str]) -> None:
        """Mark last response as good — boosts learning signal."""
        if not self.learner:
            print(_c("yellow", "  Learning not enabled")); return
        tid = self.session.last_turn_id
        if tid:
            self.learner.feedback(tid, is_good=True)
            # Re-observe with high user signal
            self.learner.observe(
                self.session.last_user,
                self.session.last_assistant,
                user_signal=1.0,
            )
        print(_c("green", "  ✓ Good feedback recorded — LionAI is learning"))

    def _cmd_bad(self, p: List[str]) -> None:
        """Mark last response as bad — negative learning signal."""
        if not self.learner:
            print(_c("yellow", "  Learning not enabled")); return
        tid = self.session.last_turn_id
        if tid:
            self.learner.feedback(tid, is_good=False)
            self.learner.observe(
                self.session.last_user,
                self.session.last_assistant,
                user_signal=0.0,
            )
        print(_c("red", "  ✗ Bad feedback recorded — LionAI will avoid this pattern"))

    def _cmd_correct(self, p: List[str]) -> None:
        """
        /correct "wrong answer" "correct answer"
        Teaches the AI what the right answer is via contrastive learning.
        """
        if not self.learner:
            print(_c("yellow", "  Learning not enabled")); return

        # Parse: /correct "OLD ANSWER" "NEW ANSWER"
        full = " ".join(p[1:])
        import re
        quoted = re.findall(r'"([^"]+)"', full)
        if len(quoted) < 2:
            # Try unquoted: /correct bad_word good_word
            parts = full.strip().split(None, 1)
            if len(parts) < 2:
                print("  Usage: /correct \"wrong answer\" \"correct answer\"")
                print("  Example: /correct \"Paris is in Germany\" \"Paris is in France\"")
                return
            bad, good = parts[0], parts[1]
        else:
            bad, good = quoted[0], quoted[1]

        prompt = self.session.last_user or "Correct this:"
        loss   = self.learner.correct(prompt, bad_response=bad, good_response=good)

        print(_c("green", f"  ✓ Correction applied (contrastive_loss={loss:.4f})"))
        print(_c("dim",   f"  Wrong:   {bad[:60]}"))
        print(_c("dim",   f"  Correct: {good[:60]}"))
        print(_c("dim",   "  LionAI will prefer the correct answer in future"))

    def _cmd_learn_stats(self, p: List[str]) -> None:
        """Show real-time learning statistics."""
        if not self.learner:
            print(_c("yellow", "  Learning not enabled (start with --lora)"))
            return
        s = self.learner.stats()
        print(_c("bold", "\n  ── Real-Time Learning Stats ──"))
        print(f"  LoRA updates:       {s['update_count']}")
        print(f"  Avg training loss:  {s['avg_train_loss']:.4f}")
        print(f"  Total turns stored: {s['total_turns']}")
        print(f"  High quality turns: {s['high_quality']}")
        print(f"  Learned turns:      {s['learned_turns']}")
        print(f"  Avg reward:         {s['avg_reward']:.3f}")
        print(f"  LoRA params:        {s['lora_params']:,}")
        print(f"  Learning enabled:   {s['lora_enabled']}")
        print()
        print(_c("dim", "  Use /good and /bad after responses to guide learning."))
        print(_c("dim", "  Use /correct to teach specific right/wrong pairs.\n"))

    def _cmd_save_lora(self, p: List[str]) -> None:
        """Manually save the current LoRA weights."""
        if not self.learner:
            print(_c("yellow", "  No learner active")); return
        path = Path(p[1]) if len(p) > 1 else self._model_dir / "lora_online"
        self.learner.save_checkpoint(path)
        print(_c("green", f"  LoRA checkpoint saved → {path}"))

    # ─── Command dispatcher ──────────────────────────────────────────────────

    def _handle_command(self, line: str) -> None:
        parts = line.strip().split(None, 2)
        cmd   = parts[0].lower()
        handler = self._cmds.get(cmd)
        if handler: handler(parts)
        else: print(_c("yellow", f"  Unknown: {cmd}  (type /help)"))

    # ─── Main Loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        print(_c("cyan", BANNER))
        learning_str = (_c("green", "● LEARNING ON") if self.learner
                        else _c("dim", "○ learning off"))
        print(_c("bold",
                 f"  LionAI v3  |  {self._device.upper()}  |  "
                 f"quant={self._quant}  |  {learning_str}"))
        print(_c("dim",
                 f"  RAM={self._hw.ram_gb:.0f}GB  "
                 f"{'AMD/ROCm  ' if self._hw.is_amd else ''}"
                 f"reasoning={'on' if self.gen_cfg.use_reasoning else 'off'}  "
                 f"|  /help for commands\n"))

        while self._running:
            try:
                user_input = input(_c("blue", "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.memory.save_session()
                if self.learner:
                    self.learner.save_checkpoint(self._model_dir / "lora_online")
                break

            if not user_input: continue
            if user_input.startswith("/"):
                self._handle_command(user_input); continue

            try:
                user_input = self._validator.validate_text(user_input)
            except ValueError as e:
                print(_c("red", f"  Rejected: {e}")); continue

            self.memory.short.add("user", user_input)

            try:
                response = self._process_turn(user_input)
            except Exception as e:
                logger.error("Generation error: %s", e, exc_info=True)
                print(_c("red", f"  [Error: {e}]")); continue

            self.memory.short.add("assistant", response)
            self.session.add("user",      user_input)
            self.session.add("assistant", response)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="LionAI — Real-Time Learning Chat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   default="./runs/lionai/final")
    parser.add_argument("--data",    default="./data")
    parser.add_argument("--device",  default=None, choices=["cpu","cuda","mps","auto"])
    parser.add_argument("--quantize",default="auto",
                        choices=["none","auto","fp16","bf16","int8","int4"])
    parser.add_argument("--mode",    default="sample",
                        choices=["sample","contrastive","beam"])
    parser.add_argument("--lora-r",  type=int, default=8,
                        help="LoRA rank for real-time learning (0=disable)")
    parser.add_argument("--no-learn",action="store_true",
                        help="Disable real-time learning")
    parser.add_argument("--no-cot",  action="store_true",
                        help="Disable chain-of-thought reasoning")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config",  default=None)
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "WARNING", log_to_file=False)

    model_path = Path(args.model)
    if not model_path.exists():
        print(_c("red", f"  Model not found: {model_path}"))
        print(_c("dim", "  Run: python demo_setup.py"))
        sys.exit(1)

    cfg   = (SystemConfig.load(Path(args.config)) if args.config
             else SystemConfig.from_hardware())
    quant = args.quantize
    if quant == "auto": quant = cfg.quantization

    bot = Chatbot(
        model_dir        = model_path,
        data_dir         = Path(args.data),
        device           = args.device,
        quantize         = quant,
        config           = cfg,
        lora_r           = args.lora_r,
        enable_learning  = not args.no_learn,
    )
    bot.gen_cfg.mode          = args.mode
    bot.gen_cfg.use_reasoning = not args.no_cot
    bot.run()


if __name__ == "__main__":
    main()
