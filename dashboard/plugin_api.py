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
def get_models():
    """Model picker source — the SAME payload as every Hermes model menu.

    build_models_payload(load_picker_context()) is exactly what the desktop
    model dropdown consumes: built-in + canonical/plugin providers (AGCs like
    antigravity-agy, kiro-acp, typesafe-jev) + user-config custom providers,
    each gated on real credentials. Sync def on purpose: FastAPI runs it in a
    threadpool and non_blocking_catalogs keeps it off live probes.
    """
    out = []
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context
        payload = build_models_payload(
            load_picker_context(),
            for_picker=True, non_blocking_catalogs=True)
        for r in payload.get("providers", []) or []:
            if not isinstance(r, dict):
                continue
            slug = str(r.get("slug") or "")
            if not slug or slug in ("council", "moa"):
                continue
            models = [str(m) for m in (r.get("models") or [])]
            if not models:
                continue
            out.append({"provider": slug,
                        "name": str(r.get("name") or slug),
                        "models": models})
    except Exception:
        pass
    return {"providers": out}


@router.get("/runs")
async def runs(limit: int = 30):
    return {"runs": _mod("runstore").list_runs(limit)}
