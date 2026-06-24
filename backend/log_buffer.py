"""Buffer em memória dos logs recentes da API (para o backoffice)."""
from __future__ import annotations

import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone

_BUFFER: deque[dict] = deque(maxlen=3000)
_LOCK = threading.Lock()
_HANDLER_ATTACHED = False


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            with _LOCK:
                _BUFFER.append(entry)
        except Exception:
            pass


def attach_memory_handler(logger: logging.Logger, formatter: logging.Formatter | None = None) -> None:
    global _HANDLER_ATTACHED
    if _HANDLER_ATTACHED:
        return
    handler = MemoryLogHandler()
    handler.setLevel(logging.INFO)
    if formatter:
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    _HANDLER_ATTACHED = True


def get_memory_logs(*, limit: int = 300, q: str = "", level: str = "") -> list[dict]:
    limit = max(1, min(int(limit or 300), 1000))
    with _LOCK:
        items = list(_BUFFER)
    if level:
        lvl = level.upper()
        items = [x for x in items if x.get("level") == lvl]
    if q:
        ql = q.strip().lower()
        if ql:
            items = [
                x
                for x in items
                if ql in (x.get("message") or "").lower()
                or ql in (x.get("logger") or "").lower()
            ]
    return items[-limit:]


def tail_log_file(path: str, *, limit: int = 200) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    limit = max(1, min(int(limit or 200), 500))
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            chunk = min(size, 256 * 1024)
            fh.seek(max(0, size - chunk))
            data = fh.read().decode("utf-8", errors="replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-limit:]
    except OSError:
        return []


def format_logs_text(items: list[dict]) -> str:
    return "\n".join(
        f"{x.get('ts', '')} {str(x.get('level', '')).ljust(7)} {x.get('message', '')}"
        for x in items
    )
