"""hermes-council desktop backend — /api/plugins/hermes-council/*.

Runs inside the gateway process; member calls borrow the user's configured
provider credentials via the engine (call_llm). Routes:
  GET  /config      — the single fixed-roster preset
  POST /config      — save seat/judge/chairman model assignments
  GET  /models      — the user's configured providers+models (picker source)
  GET  /runs        — recent deliberations
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()
_PKG = Path(__file__).resolve().parent.parent


def _mod(name: str):
    """Import a plugin module as part of the hermes_council package."""
    import importlib
    import importlib.util
    pkg_key = "hermes_council"
    if pkg_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_key, _PKG / "__init__.py", submodule_search_locations=[str(_PKG)])
        m = importlib.util.module_from_spec(spec)
        sys.modules[pkg_key] = m
        spec.loader.exec_module(m)
    return importlib.import_module(f"{pkg_key}.{name}")


@router.get("/config")
async def get_config():
    cc = _mod("council_config")
    from hermes_cli.config import load_config
    cfg = cc.normalize_council_config(load_config().get("council"))
    preset = cfg["presets"][cc.DEFAULT_COUNCIL_PRESET_NAME]
    return {"seats": cc.COUNCIL_SEATS, "preset": preset}


@router.post("/config")
async def save_config(body: dict):
    cc = _mod("council_config")
    problems = cc.validate_council_payload(body)
    if problems:
        return {"error": "; ".join(problems[:5])}
    preset = (body.get("presets") or {}).get("default") or body
    if not cc.council_config_complete(cc.normalize_council_config({"presets": {"default": preset}})["presets"]["default"]):
        return {"error": "chairman and at least one seat model are required"}
    cc.save_council_config(body)
    return {"ok": True}


@router.get("/models")
async def get_models():
    out = []
    try:
        from hermes_cli.config import load_config
        providers = (load_config() or {}).get("providers") or {}
        for slug, p in providers.items():
            if not isinstance(p, dict) or slug in ("council", "moa"):
                continue
            models = list((p.get("models") or {}).keys())
            if not models and p.get("model"):
                models = [p["model"]]
            if models:
                out.append({"provider": slug, "name": p.get("name") or slug, "models": models})
    except Exception:
        pass
    return {"providers": out}


@router.get("/runs")
async def runs(limit: int = 30):
    return {"runs": _mod("runstore").list_runs(limit)}
