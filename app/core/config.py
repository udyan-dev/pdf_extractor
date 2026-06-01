import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class Config:
    max_workers: int
    queue_size: int
    cpu_fallback: int
    base_job_timeout: float
    timeout_per_mb: float
    max_job_timeout: float
    max_file_size: int
    target_latency: float
    temp_dir: str
    small_file_priority_mb: float
    medium_file_priority_mb: float
    retry_after_floor: float
    request_id_header: str


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


load_env_file()


# ---------- Internal Parsing Utils (Zero Duplication) ----------

def _get_env(name: str) -> str | None:
    return os.environ.get(name)


def _int(name: str, default: int, *, min_val: int = 1, max_val: int | None = None) -> int:
    raw = _get_env(name)
    if raw is None:
        value = default
    else:
        value = int(raw)

    if value < min_val:
        raise ValueError(f"{name} < {min_val}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{name} > {max_val}")

    return value


def _float(name: str, default: float, *, min_val: float = 0.0) -> float:
    raw = _get_env(name)
    if raw is None:
        value = default
    else:
        value = float(raw)

    if value <= min_val:
        raise ValueError(f"{name} <= {min_val}")

    return value


def _text(name: str, default: str) -> str:
    raw = _get_env(name)
    value = default if raw is None else raw.strip()
    if not value:
        raise ValueError(f"{name} empty")
    return value


# ---------- Main Config Loader (Single Pass) ----------

def get_config() -> Config:
    env = os.environ  # local binding (faster)

    # temp dir (resolved once)
    temp_dir: Final[str] = env.get("PDF_TEMP_DIR") or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    # CPU-aware default (better than static 4)
    cpu_count = os.cpu_count() or 2
    queue_default = max(cpu_count * 8, 32)
    queue_max = max(cpu_count * 64, 128)

    return Config(
        max_workers=_int("PDF_MAX_WORKERS", cpu_count, max_val=cpu_count),
        queue_size=_int("PDF_QUEUE_SIZE", queue_default, max_val=queue_max),
        cpu_fallback=_int("PDF_CPU_FALLBACK", max(cpu_count // 2, 1)),
        base_job_timeout=_float("PDF_BASE_JOB_TIMEOUT", 15.0),
        timeout_per_mb=_float("PDF_TIMEOUT_PER_MB", 8.0),
        max_job_timeout=_float("PDF_MAX_JOB_TIMEOUT", 120.0),
        max_file_size=_int("PDF_MAX_FILE_SIZE", 16 * 1024 * 1024),
        target_latency=_float("PDF_TARGET_LATENCY", 4.0),
        temp_dir=temp_dir,
        small_file_priority_mb=_float("PDF_SMALL_FILE_PRIORITY_MB", 2.0),
        medium_file_priority_mb=_float("PDF_MEDIUM_FILE_PRIORITY_MB", 8.0),
        retry_after_floor=_float("PDF_RETRY_AFTER_FLOOR", 1.0),
        request_id_header=_text("PDF_REQUEST_ID_HEADER", "X-Request-ID"),
    )