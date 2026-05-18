from __future__ import annotations

import logging
import os
import resource

LOGGER = logging.getLogger(__name__)

GIB = 1024**3
DEFAULT_MEMORY_LIMIT_GB = 8.0


def bytes_from_gb(value: float) -> int:
    return int(float(value) * GIB)


def set_process_memory_limit(limit_bytes: int) -> None:
    """Apply a best-effort address-space limit for the current process."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = limit_bytes if soft < 0 else min(soft, limit_bytes)
        new_hard = hard if hard >= 0 else limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not apply process memory limit: %s", exc)


def current_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports bytes. This project runs in Linux
    # containers, but keep the fallback harmless for local development.
    return int(usage.ru_maxrss * 1024 if os.name == "posix" else usage.ru_maxrss)


def assert_memory_budget(limit_bytes: int, stage: str, safety_fraction: float = 0.92) -> None:
    rss = current_rss_bytes()
    if rss > int(limit_bytes * safety_fraction):
        raise MemoryError(
            f"Memory budget exceeded during {stage}: "
            f"RSS={rss / GIB:.2f} GiB, limit={limit_bytes / GIB:.2f} GiB"
        )
