"""CLI helpers for configuring Council (fork of hermes_cli/moa_cmd.py)."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import load_config, save_config
from .council_config import (
    COUNCIL_SEATS, DEFAULT_COUNCIL_PRESET_NAME,
    council_config_complete, normalize_council_config)


def _prompt_choice(title: str, rows: list[str], default: int = 0) -> int:
    try:
        from hermes_cli.curses_ui import curses_radiolist
        return curses_radiolist(title, rows, selected=default, cancel_returns=default)
    except Exception:
        for idx, row in enumerate(rows, start=1):
            print(f"{idx}. {row}")
        raw = input(f"{title} [{default + 1}]: ").strip()
        if not raw:
            return default
        try:
            return max(0, min(len(rows) - 1, int(raw) - 1))
        except ValueError:
            return default


def _model_options() -> list[dict[str, Any]]:
    from hermes_cli.inventory import build_models_payload, load_picker_context
    payload = build_models_payload(
        load_picker_context(),
        # Slot pickers must only offer providers the user can actually call.
        include_unconfigured=False,
        picker_hints=True,
        canonical_order=True,
        pricing=True,
        capabilities=True,
        max_models=200)
    providers = payload.get("providers") or []
    # Exclude the virtual providers: a council (or moa) slot naming one would recurse.
    _virtual = {"council", "moa"}
    return [p for p in providers
            if p.get("slug") and str(p.get("slug")).strip().lower() not in _virtual and p.get("models")]


def _prompt_optional(prompt: str, limit: int) -> str:
    """Optional free-text prompt: empty line = unset (returns ""). Curses-less fallback so
    the flow also works when stdin is not a TTY-attached curses screen."""
    try:
        raw = input(f"{prompt} (optional, Enter to skip): ")
    except EOFError:
        return ""
    return str(raw or "").strip()[:limit]


def _pick_slot(current: dict[str, str] | None = None) -> dict[str, str]:
    providers = _model_options()
    if not providers:
        raise RuntimeError("No configured model providers found. Run `hermes model` first.")
    current_provider = (current or {}).get("provider", "")
    provider_default = next((idx for idx, p in enumerate(providers) if p.get("slug") == current_provider), 0)
    provider_rows = [f"{p.get('name') or p.get('slug')}  ({p.get('slug')})" for p in providers]
    provider = providers[_prompt_choice("Select provider", provider_rows, provider_default)]
    models = list(provider.get("models") or [])
    if not models:
        raise RuntimeError(f"Provider {provider.get('slug')} has no selectable models")
    current_model = (current or {}).get("model", "")
    model_default = models.index(current_model) if current_model in models else 0
    model = models[_prompt_choice(f"Select model for {provider.get('slug')}", models, model_default)]
    return {"provider": str(provider.get("slug") or ""), "model": str(model)}


def _format_slot(slot: dict[str, Any] | None) -> str:
    if not isinstance(slot, dict) or not slot.get("provider"):
        return "(unset)"
    label = f"{slot.get('provider')}:{slot.get('model')}"
    effort = str(slot.get("reasoning_effort") or "").strip()
    return f"{label} [reasoning={effort}]" if effort else label


def _provider_mismatch_notice(cfg: dict[str, Any], chairman: dict[str, Any]) -> str | None:
    main_provider = ""
    if isinstance(cfg, dict):
        model_section = cfg.get("model")
        if isinstance(model_section, dict):
            main_provider = str(model_section.get("provider") or "").strip().lower()
    chair_provider = str((chairman or {}).get("provider") or "").strip().lower()
    if (
        not main_provider
        or not chair_provider
        # "auto" is a routing pseudo-provider, not a billing seat; the virtual providers likewise.
        or main_provider in ("council", "moa", "auto")
        or main_provider == chair_provider
    ):
        return None
    return (
        f"Chairman is on {chair_provider}; the whole tool loop will be billed there, "
        f"not to {main_provider}."
    )


def _print_config(config: dict[str, Any]) -> None:
    cfg = _council_section(config)
    preset = cfg["presets"][DEFAULT_COUNCIL_PRESET_NAME]
    print("Council (single fixed roster)")
    print("  Seats (pick a model for each):")
    for idx, slot in enumerate(preset["members"], start=1):
        seat = str(slot.get("name") or f"Seat {idx}")
        state = _format_slot(slot)
        if not slot.get("enabled", True) and slot.get("provider"):
            state += " (disabled)"
        print(f"    {idx}. {seat}  ->  {state}")
        role = str(slot.get("role") or "").strip()
        if role:
            print(f"       {role}")
    judge = preset.get("judge") or {}
    if judge.get("provider"):
        judge_state = "" if judge.get("enabled", True) else " (disabled)"
        print(f"  Judge: {_format_slot(judge)}{judge_state} (ranks the anonymized member answers)")
    else:
        print("  Judge: (unset — members review each other anonymously)")
    chair_slot = preset.get("chairman") or {}
    print(
        f"  Chairman: {_format_slot(chair_slot)} "
        "(acting model — runs every step and carries almost all of the cost)"
    )
    if preset.get("lite"):
        print("  Lite: on (stage-2 review skipped)")
    notice = _provider_mismatch_notice(config, chair_slot)
    if notice:
        print(f"    note: {notice}")


def _council_section(cfg: Any) -> dict[str, Any]:
    return normalize_council_config(cfg.get("council") if isinstance(cfg, dict) else {})


def _save(cfg: dict, council: dict[str, Any]) -> None:
    cfg["council"] = normalize_council_config(council)
    save_config(cfg)


def _cmd_list(cfg: dict, args) -> None:
    _print_config(cfg)


def _cmd_configure(cfg: dict, args) -> None:
    council = _council_section(cfg)
    preset_name = DEFAULT_COUNCIL_PRESET_NAME
    current: dict[str, Any] = council["presets"][preset_name]
    print("Configure the Council (4 fixed seats). Pick a model for each (Enter keeps the current one).")
    existing = list(current.get("members") or [])
    members: list[dict[str, Any]] = []
    for idx, seat in enumerate(COUNCIL_SEATS):
        base = existing[idx] if idx < len(existing) else {}
        current_model = _format_slot(base)
        print(f"\n[{idx + 1}/4] {seat['name']} — {seat['role']}")
        print(f"      current: {current_model}")
        if _prompt_choice(f"Seat: {seat['name']}", ["Keep current", "Pick a model", "Leave unassigned"],
                          0 if base.get("provider") else 1) == 1:
            picked = _pick_slot({k: base.get(k, "") for k in ("provider", "model")})
            members.append({"provider": picked["provider"], "model": picked["model"], "enabled": True})
        elif base.get("provider"):
            members.append({"provider": base["provider"], "model": base["model"],
                            "enabled": bool(base.get("enabled", True))})
        else:
            members.append({"provider": "", "model": "", "enabled": False})
    # Judge slot is optional: empty means peer-review mode (every member reviews the others).
    print("\nConfigure the judge model (optional).")
    print(
        "The judge ranks the anonymized member answers in one call. Leave it unset to have the "
        "members review each other instead (peer review)."
    )
    current["members"] = members
    judge_rows = ["(unset — members review each other)", "Pick a judge model"]
    current_judge = current.get("judge") or {}
    judge_default = 1 if current_judge.get("provider") else 0
    if _prompt_choice("Judge slot", judge_rows, judge_default) == 1:
        picked = _pick_slot({k: current_judge.get(k, "") for k in ("provider", "model")})
        current["judge"] = {**picked, "enabled": bool(current_judge.get("enabled", True))}
    else:
        current["judge"] = {"provider": "", "model": "", "enabled": bool(current_judge.get("enabled", True))}
    print("\nConfigure the chairman model.")
    print(
        "The chairman is the acting model: it runs every tool-loop step, and almost all of the "
        "run's cost lands on its provider. Members answer once per user turn and review each "
        "other (or the judge ranks), so a council run bills N+1 (lite) up to 2N+1 model calls "
        "per turn on top of the chairman."
    )
    current["chairman"] = _pick_slot(current.get("chairman"))
    if not council_config_complete(current):
        raise SystemExit(
            "Refusing to save an incomplete Council preset: the chairman slot and at least one "
            "assigned member seat are required.")
    council["presets"][preset_name] = current
    notice = _provider_mismatch_notice(cfg, current["chairman"])
    if notice:
        print(notice)
    _save(cfg, council)
    print("Saved Council configuration.")
    _print_config(cfg)


_SUBCOMMANDS = {
    "list": _cmd_list,
    "ls": _cmd_list,
    "config": _cmd_configure,
    "configure": _cmd_configure}


def cmd_council(args) -> None:
    """Manage Council model presets."""
    cfg = load_config()
    sub = getattr(args, "council_command", None) or "list"
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        raise SystemExit(f"Unknown council subcommand: {sub}")
    handler(cfg, args)
