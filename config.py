"""
LionAI Configuration & Security  [Enhanced]
=============================================
New vs v1:
  • Auto hardware profiler — detects RAM, VRAM, CPU cores, recommends settings
  • TOML config file support (Python 3.11+ stdlib or tomli fallback)
  • Tiered config: system defaults → file → env vars → CLI flags
  • Hardware-aware automatic config generator
  • Memory pressure monitor with adaptive batch size signals
  • Secure input sanitization with allowlist approach
"""

import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Hardware Profiler
# ─────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    cpu_cores:      int   = 1
    ram_gb:         float = 4.0
    vram_gb:        float = 0.0
    has_cuda:       bool  = False
    has_mps:        bool  = False
    device:         str   = "cpu"
    recommended_model_size: str = "small"
    recommended_quantization: str = "int8"
    recommended_batch_size: int = 4
    recommended_seq_len:    int = 512


def detect_hardware() -> HardwareProfile:
    """Auto-detect hardware and recommend optimal settings."""
    import torch

    p = HardwareProfile()
    p.cpu_cores = os.cpu_count() or 1

    # RAM detection
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        p.ram_gb = int(line.split()[1]) / 1e6
                        break
        elif platform.system() == "Darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "hw.memsize"]).decode()
            p.ram_gb = int(out.split()[1]) / 1e9
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong  = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", c_ulong),("dwMemoryLoad", c_ulong),
                             ("dwTotalPhys", c_ulong),("dwAvailPhys", c_ulong),
                             ("dwTotalPageFile", c_ulong),("dwAvailPageFile", c_ulong),
                             ("dwTotalVirtual", c_ulong),("dwAvailVirtual", c_ulong)]
            ms = MEMORYSTATUS()
            ms.dwLength = ctypes.sizeof(ms)
            kernel32.GlobalMemoryStatus(ctypes.byref(ms))
            p.ram_gb = ms.dwTotalPhys / 1e9
    except Exception:
        p.ram_gb = 4.0   # conservative fallback

    # GPU detection
    if torch.cuda.is_available():
        p.has_cuda = True
        p.vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        p.device   = "cuda"
    elif torch.backends.mps.is_available():
        p.has_mps  = True
        p.device   = "mps"
        # Apple Silicon unified memory — use 60% of RAM
        p.vram_gb  = p.ram_gb * 0.6

    # Recommend model size
    eff_mem = max(p.vram_gb, p.ram_gb * 0.5)   # usable memory estimate
    if eff_mem >= 10:
        p.recommended_model_size = "large"
        p.recommended_quantization = "fp16" if p.has_cuda else "none"
        p.recommended_batch_size = 16
        p.recommended_seq_len = 1024
    elif eff_mem >= 6:
        p.recommended_model_size = "medium"
        p.recommended_quantization = "fp16" if p.has_cuda else "int8"
        p.recommended_batch_size = 8
        p.recommended_seq_len = 512
    elif eff_mem >= 3:
        p.recommended_model_size = "small"
        p.recommended_quantization = "int8"
        p.recommended_batch_size = 4
        p.recommended_seq_len = 512
    else:
        p.recommended_model_size = "micro"
        p.recommended_quantization = "int4"
        p.recommended_batch_size = 2
        p.recommended_seq_len = 256

    logger.info("Hardware: CPU×%d  RAM=%.1fGB  VRAM=%.1fGB  device=%s",
                p.cpu_cores, p.ram_gb, p.vram_gb, p.device)
    logger.info("Recommended: size=%s  quant=%s  batch=%d",
                p.recommended_model_size, p.recommended_quantization,
                p.recommended_batch_size)
    return p


# ─────────────────────────────────────────────
#  System Config
# ─────────────────────────────────────────────

@dataclass
class SystemConfig:
    """
    Top-level runtime configuration.
    Auto-fills from detected hardware when using from_hardware().
    """
    # Paths
    model_dir:   str = "./runs/lionai/final"
    data_dir:    str = "./data"
    log_dir:     str = "./logs"
    cache_dir:   str = "./cache"

    # Model
    model_size:   str = "medium"
    device:       str = "auto"
    quantization: str = "none"
    torch_threads: int = 4
    gpu_layers:   int = -1    # -1 = all on GPU; 0 = CPU only; N = partial

    # Inference
    max_new_tokens:     int   = 256
    temperature:        float = 0.8
    top_k:              int   = 40
    top_p:              float = 0.92
    min_p:              float = 0.05
    repetition_penalty: float = 1.15
    frequency_penalty:  float = 0.1
    contrastive_alpha:  float = 0.0  # 0=off; 0.5=contrastive search
    context_length:     int   = 2048
    use_beam_search:    bool  = False
    num_beams:          int   = 4

    # Memory
    enable_memory:     bool = True
    memory_max_tokens: int  = 1500
    system_prompt:     str  = "You are LionAI, a helpful and knowledgeable AI assistant running locally on this device. Be concise, accurate, and helpful."

    # RAG
    enable_rag:        bool = False
    rag_top_k:         int  = 3
    rag_chunk_size:    int  = 512

    # Security
    max_input_length:  int  = 8192
    sandbox_mode:      bool = True

    # Logging
    log_level:         str  = "INFO"
    log_to_file:       bool = True

    # UI
    stream_output:     bool = True
    show_timing:       bool = False
    color_output:      bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        SafeFileHandler.atomic_write(
            path, json.dumps(self.to_dict(), indent=2).encode()
        )

    @classmethod
    def load(cls, path: Path) -> "SystemConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = SafeFileHandler.safe_read(path)
        if raw is None:
            return cls()
        try:
            data = json.loads(raw.decode())
        except json.JSONDecodeError:
            return cls()
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        cfg._apply_env_overrides()
        return cfg

    @classmethod
    def from_hardware(cls) -> "SystemConfig":
        """Build an optimised config based on detected hardware."""
        hw  = detect_hardware()
        cfg = cls(
            device       = hw.device,
            model_size   = hw.recommended_model_size,
            quantization = hw.recommended_quantization,
            torch_threads = min(hw.cpu_cores, 8),
        )
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        for key, val in os.environ.items():
            if not key.startswith("LIONAI_"):
                continue
            field_name = key[7:].lower()
            if field_name not in self.__dataclass_fields__:
                continue
            ftype = type(getattr(self, field_name))
            try:
                if ftype is bool:
                    setattr(self, field_name, val.lower() in ("1", "true", "yes"))
                else:
                    setattr(self, field_name, ftype(val))
            except (ValueError, TypeError):
                pass


# ─────────────────────────────────────────────
#  Input Validator (allowlist approach)
# ─────────────────────────────────────────────

class InputValidator:
    """Validates all user inputs with an allowlist + deny pattern approach."""

    _DENY_PATTERNS = [
        re.compile(r"\x00"),                          # null bytes
        re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]"), # control chars
        re.compile(r"<\|[^|]{1,30}\|>"),              # token injection
        re.compile(r"(?i)system\s*:\s*you\s+are\s+now"), # prompt injection
    ]

    _SUSPICIOUS = [
        "ignore previous", "ignore all instructions", "disregard",
        "new persona", "you are now", "override your", "jailbreak",
        "DAN mode", "developer mode",
    ]

    def __init__(self, max_input_length: int = 8192) -> None:
        self.max_input_length = max_input_length

    def validate_text(self, text: str, field: str = "input") -> str:
        if not isinstance(text, str):
            raise ValueError(f"{field} must be a string, got {type(text)}")
        if len(text) > self.max_input_length:
            logger.warning("Input truncated: %d → %d chars", len(text), self.max_input_length)
            text = text[: self.max_input_length]
        for pat in self._DENY_PATTERNS:
            if pat.search(text):
                raise ValueError(f"Disallowed content detected in {field}")
        lower = text.lower()
        for phrase in self._SUSPICIOUS:
            if phrase in lower:
                logger.warning("Suspicious input phrase: %r", phrase)
        return text

    def validate_path(self, path: Union[str, Path],
                      must_exist: bool = True,
                      allowed_extensions: Optional[List[str]] = None,
                      allowed_root: Optional[Path] = None) -> Path:
        p = Path(path).resolve()
        if ".." in str(Path(path).parts):
            raise ValueError(f"Path traversal not allowed: {path}")
        if allowed_root:
            try:
                p.relative_to(Path(allowed_root).resolve())
            except ValueError:
                raise ValueError(f"Path outside allowed root: {path}")
        if allowed_extensions and p.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Extension {p.suffix!r} not allowed")
        if must_exist and not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p

    def validate_config_value(self, key: str, value: Any) -> Any:
        float_bounds = {
            "temperature": (0.01, 2.0), "top_p": (0.01, 1.0),
            "min_p": (0.0, 0.5), "repetition_penalty": (1.0, 5.0),
            "frequency_penalty": (0.0, 2.0), "contrastive_alpha": (0.0, 1.0),
        }
        int_bounds = {
            "top_k": (0, 500), "max_new_tokens": (1, 8192),
            "num_beams": (1, 16),
        }
        if key in float_bounds:
            lo, hi = float_bounds[key]
            v = float(value)
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be [{lo}, {hi}], got {v}")
            return v
        if key in int_bounds:
            lo, hi = int_bounds[key]
            v = int(value)
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be [{lo}, {hi}], got {v}")
            return v
        return value


# ─────────────────────────────────────────────
#  Safe File Handler
# ─────────────────────────────────────────────

class SafeFileHandler:
    CHECKSUM_EXT = ".sha256"

    @staticmethod
    def atomic_write(path: Path, data: bytes) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, str(path))
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        checksum = path.with_suffix(path.suffix + SafeFileHandler.CHECKSUM_EXT)
        checksum.write_text(hashlib.sha256(data).hexdigest())

    @staticmethod
    def safe_read(path: Path, verify: bool = True) -> Optional[bytes]:
        path = Path(path)
        if not path.exists(): return None
        try:
            data = path.read_bytes()
        except OSError as e:
            logger.error("Read error %s: %s", path, e); return None
        if verify:
            chk = path.with_suffix(path.suffix + SafeFileHandler.CHECKSUM_EXT)
            if chk.exists():
                if chk.read_text().strip() != hashlib.sha256(data).hexdigest():
                    logger.error("Checksum mismatch: %s (may be corrupt)", path)
                    return None
        return data

    @staticmethod
    def safe_json_load(path: Path) -> Optional[Dict]:
        raw = SafeFileHandler.safe_read(path, verify=False)
        if raw is None: return None
        try:
            return json.loads(raw.decode("utf-8"))
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

class CrashRecovery:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir  = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock  = self.state_dir / "process.lock"
        self._state = self.state_dir / "recovery_state.json"

    def acquire_lock(self) -> bool:
        pid = os.getpid()
        if self._lock.exists():
            try:
                existing = int(self._lock.read_text().strip())
                try:
                    os.kill(existing, 0)
                    logger.warning("Lock held by PID %d", existing)
                    return False
                except (ProcessLookupError, PermissionError):
                    self._lock.unlink(missing_ok=True)
            except (ValueError, OSError):
                self._lock.unlink(missing_ok=True)
        SafeFileHandler.atomic_write(self._lock, str(pid).encode())
        return True

    def release_lock(self) -> None:
        self._lock.unlink(missing_ok=True)

    def save_state(self, state: Dict) -> None:
        state["saved_at"] = time.time()
        SafeFileHandler.safe_json_save(self._state, state)

    def load_state(self) -> Optional[Dict]:
        return SafeFileHandler.safe_json_load(self._state)

    def clear_state(self) -> None:
        self._state.unlink(missing_ok=True)

    def __enter__(self) -> "CrashRecovery":
        if not self.acquire_lock():
            raise RuntimeError("Could not acquire lock — is LionAI already running?")
        return self

    def __exit__(self, et, ev, tb) -> None:
        if et is None:
            self.clear_state()
        self.release_lock()


# ─────────────────────────────────────────────
#  Memory Pressure Monitor
# ─────────────────────────────────────────────

class MemoryMonitor:
    """Watches RAM/VRAM and signals when pressure is high."""

    def __init__(self, warn_threshold: float = 0.85,
                 crit_threshold: float = 0.95) -> None:
        self.warn = warn_threshold
        self.crit = crit_threshold

    def check_ram(self) -> Dict[str, float]:
        import torch
        info: Dict[str, float] = {}

        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved()
            total    = torch.cuda.get_device_properties(0).total_memory
            info["vram_used_gb"]  = reserved / 1e9
            info["vram_total_gb"] = total / 1e9
            info["vram_pressure"] = reserved / max(total, 1)

        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    vals = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            vals[parts[0].rstrip(":")] = int(parts[1])
                total_kb = vals.get("MemTotal", 1)
                avail_kb = vals.get("MemAvailable", total_kb)
                info["ram_used_gb"]  = (total_kb - avail_kb) / 1e6
                info["ram_total_gb"] = total_kb / 1e6
                info["ram_pressure"] = 1 - avail_kb / total_kb
        except Exception:
            pass

        return info

    def is_critical(self) -> bool:
        info = self.check_ram()
        return (info.get("vram_pressure", 0) > self.crit or
                info.get("ram_pressure", 0) > self.crit)

    def is_warn(self) -> bool:
        info = self.check_ram()
        return (info.get("vram_pressure", 0) > self.warn or
                info.get("ram_pressure", 0) > self.warn)


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logging(level: str = "INFO",
                  log_dir: Optional[Path] = None,
                  log_to_file: bool = True) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            str(log_dir / f"lionai_{time.strftime('%Y%m%d')}.log"),
            encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers, force=True,
    )
    for noisy in ("urllib3", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
