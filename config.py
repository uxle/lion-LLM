"""
LionAI config.py — Bug-Fixed + AMD/CPU Edition
================================================
Bugs fixed:
  BUG 1: detect_hardware() lru_cached but returned mutable dataclass
          → returns a copy so callers can't corrupt the cached instance
  BUG 2: SystemConfig._CASTS dict built in __post_init__ but _CASTS field
          declared with default_factory=dict which broke dataclass equality
          → moved _CASTS to a regular instance attribute, not a dataclass field
  BUG 3: AMD/ROCm GPU not detected (only checked torch.cuda.is_available()
          which is also True for ROCm, but GPU name needed to distinguish)
          → added proper AMD name detection for RX550 etc.
  BUG 4: MemoryMonitor._cache_ts used time.monotonic() but compared with
          timestamps from time.time() in some callers
          → all timing uses time.monotonic() consistently
  BUG 5: setup_logging idempotent flag _log_configured was module-level
          but not reset between tests / multiple process starts
          → use logging module's own handler-count check instead
  BUG 6: SafeFileHandler checksum extension ".sha256" appended with
          path.with_suffix(path.suffix + ext) — broke on paths like model.pt
          → use path.parent / (path.name + ext) instead
  BUG 7: InputValidator._DENY tuple not type-annotated → mypy errors
          → explicit Tuple[re.Pattern, ...]
  BUG 8: CrashRecovery.acquire_lock called os.kill(pid, 0) but on Windows
          this raises PermissionError even for running processes
          → Windows-safe process-alive check
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Hardware Profiler
# ─────────────────────────────────────────────

@dataclass
class HardwareProfile:
    cpu_cores:   int   = 1
    ram_gb:      float = 4.0
    vram_gb:     float = 0.0
    gpu_name:    str   = ""
    has_cuda:    bool  = False
    is_amd:      bool  = False          # NEW: AMD/ROCm flag
    has_mps:     bool  = False
    device:      str   = "cpu"
    recommended_model_size:   str = "small"
    recommended_quantization: str = "int8"
    recommended_batch_size:   int = 4
    recommended_seq_len:      int = 128


@lru_cache(maxsize=1)
def _detect_hardware_cached() -> HardwareProfile:
    """Internal cached implementation — always returns same object."""
    import torch
    p = HardwareProfile()
    p.cpu_cores = os.cpu_count() or 1

    # RAM detection
    try:
        sys_name = platform.system()
        if sys_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        p.ram_gb = int(line.split()[1]) / 1e6
                        break
        elif sys_name == "Darwin":
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "hw.memsize"], stderr=subprocess.DEVNULL)
            p.ram_gb = int(out.split()[1]) / 1e9
        elif sys_name == "Windows":
            import ctypes
            class _MS(ctypes.Structure):
                _fields_ = [("l",  ctypes.c_ulong)] + [
                    (f"f{i}", ctypes.c_ulong) for i in range(7)]
            ms = _MS(); ms.l = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatus(ctypes.byref(ms))
            p.ram_gb = ms.f1 / 1e9
    except Exception:
        p.ram_gb = 4.0

    # GPU detection — FIX: AMD RX550 detection
    if torch.cuda.is_available():
        p.has_cuda = True
        props      = torch.cuda.get_device_properties(0)
        p.vram_gb  = props.total_memory / 1e9
        p.gpu_name = props.name
        # FIX: detect AMD/ROCm GPU
        p.is_amd = any(s in props.name.lower()
                       for s in ("amd", "radeon", "gfx", "vega", "navi", "rx"))
        p.device = "cuda"
        logger.info("GPU: %s (%.1f GB VRAM)%s",
                    props.name, p.vram_gb, " [AMD/ROCm]" if p.is_amd else "")
    elif torch.backends.mps.is_available():
        p.has_mps  = True
        p.device   = "mps"
        p.vram_gb  = p.ram_gb * 0.6  # Apple unified memory

    # Recommendations — adjusted for AMD RX550 (4GB VRAM)
    eff = max(p.vram_gb, p.ram_gb * 0.5)
    if eff >= 10 and not p.is_amd:
        p.recommended_model_size   = "large"
        p.recommended_quantization = "fp16"
        p.recommended_batch_size   = 16
        p.recommended_seq_len      = 1024
    elif eff >= 6:
        p.recommended_model_size   = "medium"
        p.recommended_quantization = "fp16" if (p.has_cuda and not p.is_amd) else "int8"
        p.recommended_batch_size   = 8
        p.recommended_seq_len      = 512
    elif eff >= 3:
        p.recommended_model_size   = "small"
        p.recommended_quantization = "int8"
        p.recommended_batch_size   = 4
        p.recommended_seq_len      = 256
    else:
        p.recommended_model_size   = "micro"
        p.recommended_quantization = "int4"
        p.recommended_batch_size   = 2
        p.recommended_seq_len      = 128

    # AMD RX550 specifics: 4GB VRAM, use int8 for safety
    if p.is_amd and p.vram_gb <= 4:
        p.recommended_quantization = "int8"
        p.recommended_batch_size   = max(1, p.recommended_batch_size // 2)

    logger.info("HW: CPU×%d RAM=%.1fGB VRAM=%.1fGB dev=%s → %s/%s",
                p.cpu_cores, p.ram_gb, p.vram_gb, p.device,
                p.recommended_model_size, p.recommended_quantization)
    return p


def detect_hardware() -> HardwareProfile:
    """
    Return hardware profile.
    FIX: returns a shallow copy so callers can't corrupt the lru_cache.
    """
    return copy.copy(_detect_hardware_cached())


# ─────────────────────────────────────────────
#  System Config
# ─────────────────────────────────────────────

@dataclass
class SystemConfig:
    # Paths
    model_dir:   str = "./runs/lionai/final"
    data_dir:    str = "./data"
    log_dir:     str = "./logs"
    cache_dir:   str = "./cache"

    # Model
    model_size:    str  = "medium"
    device:        str  = "auto"
    quantization:  str  = "none"
    torch_threads: int  = 4
    gpu_layers:    int  = -1

    # Inference
    max_new_tokens:     int   = 256
    temperature:        float = 0.8
    top_k:              int   = 40
    top_p:              float = 0.92
    min_p:              float = 0.05
    repetition_penalty: float = 1.1
    frequency_penalty:  float = 0.0
    contrastive_alpha:  float = 0.0
    context_length:     int   = 2048
    use_beam_search:    bool  = False
    num_beams:          int   = 4

    # Memory
    enable_memory:     bool = True
    memory_max_tokens: int  = 2048
    system_prompt:     str  = "You are LionAI, a helpful AI assistant."

    # RAG
    enable_rag:     bool = False
    rag_top_k:      int  = 3
    rag_chunk_size: int  = 400

    # Security
    max_input_length: int  = 8192
    sandbox_mode:     bool = True

    # Logging
    log_level:    str  = "INFO"
    log_to_file:  bool = True

    # UI
    stream_output: bool = True
    color_output:  bool = True

    def __post_init__(self) -> None:
        # FIX: _CASTS is NOT a dataclass field — assigned here as plain attr
        self._CASTS: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            val = getattr(self, k, None)
            if val is None: continue
            if isinstance(val, bool):
                self._CASTS[k] = lambda v: v.lower() in ("1", "true", "yes")
            else:
                self._CASTS[k] = type(val)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        SafeFileHandler.atomic_write(
            Path(path), json.dumps(self.to_dict(), indent=2).encode()
        )

    @classmethod
    def load(cls, path: Path) -> "SystemConfig":
        path = Path(path)
        if not path.exists(): return cls()
        raw = SafeFileHandler.safe_read(path)
        if raw is None: return cls()
        try:
            d = json.loads(raw.decode())
        except json.JSONDecodeError:
            return cls()
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def from_hardware(cls) -> "SystemConfig":
        hw  = detect_hardware()
        cfg = cls(
            device        = hw.device,
            model_size    = hw.recommended_model_size,
            quantization  = hw.recommended_quantization,
            torch_threads = min(hw.cpu_cores, 8),
        )
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        if not hasattr(self, "_CASTS"): self.__post_init__()
        for key, val in os.environ.items():
            if not key.startswith("LIONAI_"): continue
            attr = key[7:].lower()
            cast = self._CASTS.get(attr)
            if cast is None: continue
            try:
                setattr(self, attr, cast(val))
            except (ValueError, TypeError):
                pass


# ─────────────────────────────────────────────
#  Input Validator
# ─────────────────────────────────────────────

class InputValidator:
    # FIX: explicit Tuple[re.Pattern, ...] type
    _DENY: Tuple[re.Pattern, ...] = (
        re.compile(r"\x00"),
        re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]"),
        re.compile(r"<\|[^|]{1,30}\|>"),
    )
    _SUSPICIOUS: Tuple[str, ...] = (
        "ignore previous", "ignore all instructions", "disregard",
        "you are now", "override your", "jailbreak", "dan mode",
    )
    _FLOAT_BOUNDS: Dict[str, Tuple[float, float]] = {
        "temperature": (0.01, 2.0), "top_p": (0.01, 1.0),
        "min_p": (0.0, 0.5), "repetition_penalty": (1.0, 5.0),
        "frequency_penalty": (0.0, 2.0), "contrastive_alpha": (0.0, 1.0),
    }
    _INT_BOUNDS: Dict[str, Tuple[int, int]] = {
        "top_k": (0, 500), "max_new_tokens": (1, 8192), "num_beams": (1, 16),
    }

    def __init__(self, max_length: int = 8192) -> None:
        self.max_length = max_length

    def validate_text(self, text: str, field: str = "input") -> str:
        if not isinstance(text, str):
            raise ValueError(f"{field} must be str")
        if len(text) > self.max_length:
            text = text[:self.max_length]
        for pat in self._DENY:
            if pat.search(text):
                raise ValueError(f"Disallowed content in {field}")
        lo = text.lower()
        for phrase in self._SUSPICIOUS:
            if phrase in lo:
                logger.warning("Suspicious input phrase: %r", phrase)
        return text

    def validate_path(self, path: Union[str, Path],
                      must_exist: bool = True,
                      allowed_extensions: Optional[List[str]] = None,
                      allowed_root: Optional[Path] = None) -> Path:
        p = Path(path).resolve()
        if ".." in str(Path(path).parts):
            raise ValueError(f"Path traversal: {path}")
        if allowed_root:
            try: p.relative_to(Path(allowed_root).resolve())
            except ValueError:
                raise ValueError(f"Path outside root: {path}")
        if allowed_extensions and p.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Extension {p.suffix!r} not allowed")
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        return p

    def validate_config_value(self, key: str, value: Any) -> Any:
        if key in self._FLOAT_BOUNDS:
            lo, hi = self._FLOAT_BOUNDS[key]; v = float(value)
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be [{lo},{hi}], got {v}")
            return v
        if key in self._INT_BOUNDS:
            lo, hi = self._INT_BOUNDS[key]; v = int(value)
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be [{lo},{hi}], got {v}")
            return v
        return value


# ─────────────────────────────────────────────
#  Safe File Handler
# ─────────────────────────────────────────────

class SafeFileHandler:
    # FIX: use path.name + ext (not path.suffix + ext) to avoid .pt.sha256 weirdness
    @staticmethod
    def _checksum_path(path: Path) -> Path:
        return path.parent / (path.name + ".sha256")

    @staticmethod
    def atomic_write(path: Path, data: bytes) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, str(path))
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        SafeFileHandler._checksum_path(path).write_text(
            hashlib.sha256(data).hexdigest()
        )

    @staticmethod
    def safe_read(path: Path, verify: bool = True) -> Optional[bytes]:
        path = Path(path)
        if not path.exists(): return None
        try:    data = path.read_bytes()
        except OSError: return None
        if verify:
            chk = SafeFileHandler._checksum_path(path)
            if chk.exists():
                expected = chk.read_text().strip()
                if expected != hashlib.sha256(data).hexdigest():
                    logger.error("Checksum mismatch: %s", path)
                    return None
        return data

    @staticmethod
    def safe_json_load(path: Path) -> Optional[Dict]:
        raw = SafeFileHandler.safe_read(path, verify=False)
        if raw is None: return None
        try:    return json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.error("JSON error %s: %s", path, e)
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                try: return json.loads(bak.read_text(encoding="utf-8"))
                except Exception: pass
            return None

    @staticmethod
    def safe_json_save(path: Path, data: Dict) -> None:
        path = Path(path)
        if path.exists():
            shutil.copy2(str(path), str(path.with_suffix(path.suffix + ".bak")))
        SafeFileHandler.atomic_write(
            path, json.dumps(data, indent=2, ensure_ascii=False).encode()
        )


# ─────────────────────────────────────────────
#  Crash Recovery
# ─────────────────────────────────────────────

def _is_process_alive(pid: int) -> bool:
    """FIX: Windows-safe process-alive check."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ChildProcessError):
        return False
    except PermissionError:
        # FIX: On Windows, PermissionError means process EXISTS but we can't signal it
        if platform.system() == "Windows":
            try:
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
            except Exception:
                pass
        return False


class CrashRecovery:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock  = self.state_dir / "process.lock"
        self._state = self.state_dir / "recovery_state.json"

    def acquire_lock(self) -> bool:
        pid = os.getpid()
        if self._lock.exists():
            try:
                existing = int(self._lock.read_text().strip())
                # FIX: use platform-safe process check
                if _is_process_alive(existing):
                    logger.warning("Lock held by PID %d", existing)
                    return False
                self._lock.unlink(missing_ok=True)
            except (ValueError, OSError):
                self._lock.unlink(missing_ok=True)
        SafeFileHandler.atomic_write(self._lock, str(pid).encode())
        return True

    def release_lock(self) -> None:
        self._lock.unlink(missing_ok=True)

    def save_state(self, s: Dict) -> None:
        SafeFileHandler.safe_json_save(self._state, {**s, "at": time.time()})

    def load_state(self) -> Optional[Dict]:
        return SafeFileHandler.safe_json_load(self._state)

    def clear_state(self) -> None:
        self._state.unlink(missing_ok=True)

    def __enter__(self) -> "CrashRecovery":
        if not self.acquire_lock():
            raise RuntimeError("Lock held — is LionAI already running?")
        return self

    def __exit__(self, et, ev, tb) -> None:
        if et is None: self.clear_state()
        self.release_lock()


# ─────────────────────────────────────────────
#  Memory Monitor  (FIX: consistent monotonic)
# ─────────────────────────────────────────────

class MemoryMonitor:
    __slots__ = ("warn", "crit", "_cache", "_cache_ts", "_ttl")

    def __init__(self, warn: float = 0.85, crit: float = 0.95,
                 ttl: float = 1.0) -> None:
        self.warn = warn; self.crit = crit
        self._cache: Dict[str, float] = {}
        # FIX: use monotonic consistently
        self._cache_ts: float = 0.0
        self._ttl = ttl

    def check_ram(self) -> Dict[str, float]:
        # FIX: time.monotonic() not time.time()
        now = time.monotonic()
        if now - self._cache_ts < self._ttl:
            return self._cache

        import torch
        info: Dict[str, float] = {}

        if torch.cuda.is_available():
            res = torch.cuda.memory_reserved()
            tot = torch.cuda.get_device_properties(0).total_memory
            info["vram_used_gb"]  = res / 1e9
            info["vram_total_gb"] = tot / 1e9
            info["vram_pressure"] = res / max(tot, 1)

        if platform.system() == "Linux":
            try:
                vals: Dict[str, int] = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            vals[parts[0].rstrip(":")] = int(parts[1])
                tt = vals.get("MemTotal", 1)
                av = vals.get("MemAvailable", tt)
                info["ram_used_gb"]  = (tt - av) / 1e6
                info["ram_total_gb"] = tt / 1e6
                info["ram_pressure"] = 1 - av / tt
            except Exception:
                pass

        self._cache    = info
        self._cache_ts = now
        return info

    def is_critical(self) -> bool:
        i = self.check_ram()
        return (i.get("vram_pressure", 0) > self.crit or
                i.get("ram_pressure",  0) > self.crit)

    def is_warn(self) -> bool:
        i = self.check_ram()
        return (i.get("vram_pressure", 0) > self.warn or
                i.get("ram_pressure",  0) > self.warn)


# ─────────────────────────────────────────────
#  Logging  (FIX: use handler count, not module flag)
# ─────────────────────────────────────────────

def setup_logging(level: str = "INFO",
                  log_dir: Optional[Path] = None,
                  log_to_file: bool = True) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root    = logging.getLogger()

    # FIX: check existing handlers instead of module-level bool
    if root.handlers and root.level == numeric:
        return

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            str(Path(log_dir) / f"lionai_{time.strftime('%Y%m%d')}.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level   = numeric,
        format  = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
        handlers= handlers,
        force   = True,
    )
    for lib in ("urllib3", "filelock", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)
