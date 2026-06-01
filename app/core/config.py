import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Config:
    parse_concurrency: int
    queue_size: int
    temp_dir: str


def _read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(name)
    return parsed


@lru_cache(maxsize=1)
def get_config() -> Config:
    temp_dir = os.getenv("PDF_TEMP_DIR") or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return Config(
        parse_concurrency=_read_positive_int("PDF_PARSE_CONCURRENCY", 2),
        queue_size=_read_positive_int("PDF_QUEUE_SIZE", 16),
        temp_dir=temp_dir,
    )
