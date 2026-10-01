"""Minimal run archive for council deliberations: ~/.hermes/council/runs/."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        home = Path.home() / ".hermes"
    d = home / "council" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_deliberation(question: str, answer: str, chairman: Dict[str, Any]) -> str:
    """Persist one tool/slash deliberation; returns the run id ('' on failure)."""
    run_id = uuid.uuid4().hex[:12]
    try:
        record = {
            "run_id": run_id,
            "query": question[:2000],
            "answer": answer,
            "chairman": chairman,
            "ts": time.time(),
            "source": "tool",
        }
        with _LOCK:
            (_dir() / f"{run_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8")
            idx_path = _dir() / "index.json"
            try:
                idx = json.loads(idx_path.read_text(encoding="utf-8"))
            except Exception:
                idx = []
            idx.insert(0, {"run_id": run_id, "query": question[:200], "ts": record["ts"]})
            idx_path.write_text(json.dumps(idx[:100], ensure_ascii=False), encoding="utf-8")
        return run_id
    except Exception:
        logger.debug("council runstore save failed", exc_info=True)
        return ""


def list_runs(limit: int = 30) -> List[Dict[str, Any]]:
    try:
        return json.loads((_dir() / "index.json").read_text(encoding="utf-8"))[:max(1, min(limit, 100))]
    except Exception:
        return []


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    try:
        p = _dir() / f"{run_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except Exception:
        return None
