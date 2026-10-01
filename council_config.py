"""Council (LLM Council) configuration helpers.

Slim fork of ``hermes_cli.moa_config`` for the Council virtual provider: named presets of
member models that answer in parallel and review each other (or a dedicated judge ranks),
with a chairman that acts after the review. There is deliberately NO one-shot ``/council``
marker/decode machinery (contract section 6): Council is a session-long model selection
like a MoA preset picked from the model menu.

Read-time tolerance mirrors MoA: a hand-edited config degrades to defaults instead of
crashing; API write paths call ``validate_council_payload`` first and reject loudly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hermes_cli.moa_config import (  # shared coercion helpers (same package, MoA untouched)
    _clean_reasoning_effort,
    _coerce_bool,
    _coerce_number,
)

DEFAULT_COUNCIL_PRESET_NAME = "default"


class CouncilPresetNotFound(ValueError):
    """Plugin-local: requested council preset does not exist."""

# The FIXED council roster. Seat names and duties are product, not user config:
# the panel renders exactly these four seats and the user only picks the model
# for each. normalize forces every preset onto this roster positionally, so an
# older/hand-edited config can never rename, reorder, add or drop a seat.
COUNCIL_SEATS: tuple[dict[str, str], ...] = (
    {
        "name": "First-Principles Analyst",
        "role": "Reason from fundamentals; ignore convention and popularity.",
    },
    {
        "name": "Skeptic",
        "role": ("Assume the obvious answer is wrong; hunt for failure modes, "
                 "hidden costs and base rates."),
    },
    {
        "name": "Pragmatist",
        "role": ("What actually works in practice - constraints, effort, "
                 "reversibility, second-order effects."),
    },
    {
        "name": "Researcher",
        "role": ("Ground claims in verifiable evidence; use web search and code "
                 "inspection where available."),
    },
)

COUNCIL_SEAT_COUNT = len(COUNCIL_SEATS)
_SEAT_NAMES = tuple(seat["name"] for seat in COUNCIL_SEATS)
_SEAT_ROLES = tuple(seat["role"] for seat in COUNCIL_SEATS)

DEFAULT_COUNCIL_MEMBERS: list[dict[str, str]] = []

# An unset judge (empty provider+model) means peer-review mode: every member reviews the
# anonymized answers of the others.
DEFAULT_COUNCIL_JUDGE: dict[str, Any] = {"provider": "", "model": "", "enabled": True}

DEFAULT_COUNCIL_CHAIRMAN: dict[str, str] = {"provider": "", "model": ""}

# Virtual-provider slugs that must never appear inside a preset slot (recursive fan-out).
_FORBIDDEN_SLOT_PROVIDERS = {"council", "moa"}


def _default_members() -> list[dict[str, Any]]:
    """The four fixed seats, all unassigned (empty provider/model, disabled).

    Unassigned seats are skipped by the engine, so a fresh preset never silently
    bills a model the user did not pick."""
    return [{"provider": "", "model": "", "enabled": False,
             "name": name, "role": role}
            for name, role in zip(_SEAT_NAMES, _SEAT_ROLES)]


def slot_problem(slot: Any) -> str | None:
    """Human-readable problem for a slot the cleaner would drop; None when complete and valid.

    Mirrors ``moa_config._slot_problem`` (write-boundary validator and tolerant runtime
    normalizer can never disagree), extended to reject BOTH virtual providers: a council slot
    naming ``council`` would recurse, and ``moa`` would nest one fan-out inside another."""
    if not isinstance(slot, dict):
        return "must be an object with 'provider' and 'model'"
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    if not provider and not model:
        return "provider and model are required"
    if not provider:
        return "provider is required"
    if not model:
        return f"model is required (provider '{provider}' has no model selected)"
    if provider.lower() in _FORBIDDEN_SLOT_PROVIDERS:
        return ("the Council/MoA virtual providers cannot be used inside a preset "
                "(recursive fan-out)")
    return None


def _clean_optional_text(value: Any, limit: int) -> str | None:
    """Optional display string (``name``/``role``): stripped, clipped to ``limit`` chars.

    ``None`` means unset (the key is dropped from the clean slot). Non-strings degrade to
    unset at READ time; ``validate_council_payload`` rejects them at WRITE time."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:limit].rstrip() if text else None


# Caps for the optional per-slot display strings (name shown in the UI, role injected
# into the member's stage-1 prompt).
_NAME_MAX_CHARS = 60
_ROLE_MAX_CHARS = 500


def _clean_slot(slot: Any, *, include_enabled: bool = False, include_role: bool = False) -> dict[str, Any] | None:
    # Any slot ``slot_problem`` rejects is dropped, falling back to the preset's defaults.
    if slot_problem(slot) is not None:
        return None
    clean: dict[str, Any] = {"provider": str(slot["provider"]).strip(), "model": str(slot["model"]).strip()}
    effort = _clean_reasoning_effort(slot.get("reasoning_effort"))
    if effort:
        clean["reasoning_effort"] = effort
    name = _clean_optional_text(slot.get("name"), _NAME_MAX_CHARS)
    if name:
        clean["name"] = name
    if include_role:
        role = _clean_optional_text(slot.get("role"), _ROLE_MAX_CHARS)
        if role:
            clean["role"] = role
    if include_enabled:
        clean["enabled"] = _coerce_bool(slot.get("enabled"), True)
    return clean


def _member_slots(raw_members: Any) -> list:
    """``members`` as a list: a single mapping is wrapped, bad types degrade to ``[]``."""
    if not isinstance(raw_members, list):
        raw_members = [raw_members] if isinstance(raw_members, dict) else []
    return raw_members


def _clean_judge(raw: Any) -> dict[str, Any]:
    """Judge slot: fully empty => peer-review mode; a half-filled or invalid judge degrades
    to unset (read-time tolerance) rather than to a hardcoded paid model."""
    base = dict(DEFAULT_COUNCIL_JUDGE)
    if not isinstance(raw, dict):
        return base
    enabled = _coerce_bool(raw.get("enabled"), True)
    name = _clean_optional_text(raw.get("name"), _NAME_MAX_CHARS)
    base_named = {**base, "name": name} if name else base
    provider = str(raw.get("provider") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not provider and not model:
        return {**base_named, "enabled": enabled}
    clean = _clean_slot({"provider": provider, "model": model,
                         "reasoning_effort": raw.get("reasoning_effort"), "name": raw.get("name")})
    if clean is None:
        return {**base_named, "enabled": enabled}
    return {**clean, "enabled": enabled}


def _clean_member_slot(raw: Any, seat: dict[str, str]) -> dict[str, Any]:
    """One fixed seat, cleaned. Name/role come from COUNCIL_SEATS and are never
    taken from the config. An incomplete or invalid model assignment degrades to
    an unassigned seat (skipped by the engine) instead of being dropped."""
    slot = raw if isinstance(raw, dict) else {}
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    assigned = (
        bool(provider) and bool(model)
        and provider.lower() not in _FORBIDDEN_SLOT_PROVIDERS
    )
    clean: dict[str, Any] = {
        "provider": provider if assigned else "",
        "model": model if assigned else "",
        "enabled": _coerce_bool(slot.get("enabled"), True) if assigned else False,
        "name": seat["name"],
        "role": seat["role"],
    }
    effort = _clean_reasoning_effort(slot.get("reasoning_effort"))
    if effort and assigned:
        clean["reasoning_effort"] = effort
    return clean


def _normalize_preset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    raw_members = _member_slots(raw.get("members"))
    # Force the fixed roster positionally: seat i always takes COUNCIL_SEATS[i]'s
    # name/role, and the i-th configured member's provider/model/enabled. A config
    # with fewer members pads with unassigned seats; extra members are dropped.
    members = [
        _clean_member_slot(raw_members[i] if i < len(raw_members) else None, seat)
        for i, seat in enumerate(COUNCIL_SEATS)
    ]
    return {
        "enabled": _coerce_bool(raw.get("enabled"), True),
        "members": members,
        "judge": _clean_judge(raw.get("judge")),
        "chairman": _clean_slot(raw.get("chairman")) or deepcopy(DEFAULT_COUNCIL_CHAIRMAN),
        "lite": _coerce_bool(raw.get("lite"), False),
        # None means 'don't send it — provider default applies'.
        "member_temperature": _coerce_number(raw.get("member_temperature"), float),
        "chairman_temperature": _coerce_number(raw.get("chairman_temperature"), float)}


_FLAT_PRESET_KEYS = ("members", "judge", "chairman", "lite", "enabled")


def normalize_council_config(raw: Any) -> dict[str, Any]:
    """Return the Council config collapsed to ONE preset named ``default``.

    The Council is a single fixed roster (4 seats + judge + chairman) — there is
    no preset selector in the UI and it is always enabled. Any stored preset is
    folded into the single ``default`` preset (the raw ``default_preset`` when
    present, else the first), so legacy multi-preset configs migrate cleanly.
    """
    if not isinstance(raw, dict):
        raw = {}

    presets_raw = raw.get("presets")
    presets: dict[str, dict[str, Any]] = {}
    if isinstance(presets_raw, dict):
        for name, preset in presets_raw.items():
            clean_name = str(name or "").strip()
            if clean_name:
                presets[clean_name] = _normalize_preset(preset)
    if not presets:  # Legacy flat config becomes the default preset.
        presets[DEFAULT_COUNCIL_PRESET_NAME] = _normalize_preset(raw)

    # Collapse to a single 'default' preset (always enabled).
    wanted = str(raw.get("default_preset") or "").strip()
    source = presets.get(wanted) or next(iter(presets.values()))
    source = {**source, "enabled": True}
    presets = {DEFAULT_COUNCIL_PRESET_NAME: source}

    return {
        "default_preset": DEFAULT_COUNCIL_PRESET_NAME,
        "active_preset": "",
        "presets": presets,
        # Compatibility/flattened view of the single preset for dashboard/desktop callers.
        **{key: deepcopy(presets[DEFAULT_COUNCIL_PRESET_NAME][key]) for key in _FLAT_PRESET_KEYS}}


def council_config_complete(preset: Any) -> bool:
    """True when a preset is selectable/saveable: chairman slot set AND >= 1 enabled member.

    Used by the PUT /api/model/council write boundary and the CLI configure flow; an
    incomplete preset means the chairman would have nothing to act on (or nobody to pay)."""
    if not isinstance(preset, dict):
        return False
    chairman = preset.get("chairman")
    if not isinstance(chairman, dict):
        return False
    if not str(chairman.get("provider") or "").strip() or not str(chairman.get("model") or "").strip():
        return False
    members = preset.get("members")
    if not isinstance(members, list):
        return False
    return any(
        isinstance(m, dict) and str(m.get("provider") or "").strip() and str(m.get("model") or "").strip()
        and _coerce_bool(m.get("enabled"), True)
        for m in members)


def _optional_text_problems(slot: dict[str, Any], *, allow_role: bool = False) -> list[str]:
    """Write-boundary checks for the optional display strings: a non-string ``name`` (or
    ``role`` on member slots) is rejected instead of silently dropped by the cleaner."""
    problems = []
    for key in ("name", "role") if allow_role else ("name",):
        value = slot.get(key)
        if key in slot and value is not None and not isinstance(value, str):
            problems.append(f"{key} must be a string")
    return problems


def validate_council_payload(raw: Any) -> list[str]:
    """Return the problems ``normalize_council_config`` would silently paper over (empty = safe to save).

    Same reject-don't-repair contract as ``validate_moa_payload``: read-time tolerance is a
    corruption engine at write time."""
    if not isinstance(raw, dict):
        return ["Council config must be an object"]

    presets_raw = raw.get("presets")
    # Legacy flat payload: the top-level object is the default preset.
    presets: dict[Any, Any] = presets_raw if isinstance(presets_raw, dict) and presets_raw else {DEFAULT_COUNCIL_PRESET_NAME: raw}

    problems: list[str] = []
    for name, preset in presets.items():
        label = str(name or "").strip() or "(unnamed)"
        if not isinstance(preset, dict):
            problems.append(f"preset '{label}': must be an object")
            continue

        members = preset.get("members")
        if not isinstance(members, list):
            members = [members] if isinstance(members, dict) else []
        if len(members) > COUNCIL_SEAT_COUNT:
            problems.append(
                f"preset '{label}': the council has {COUNCIL_SEAT_COUNT} fixed seats "
                f"({', '.join(_SEAT_NAMES)}); got {len(members)} members — seats cannot be added")
        issues = []
        for index, slot in enumerate(members):
            # An unassigned seat (no provider AND no model) is legal — the engine
            # skips it. Only a half-filled or invalid assignment is a problem.
            if isinstance(slot, dict):
                provider = str(slot.get("provider") or "").strip()
                model = str(slot.get("model") or "").strip()
                if not provider and not model:
                    issues.append((index, None))
                    continue
            issues.append((index, slot_problem(slot)))
        problems.extend(f"preset '{label}' member {index + 1}: {issue}" for index, issue in issues if issue)
        for index, slot in enumerate(members):
            if not isinstance(slot, dict):
                continue
            problems.extend(
                f"preset '{label}' member {index + 1}: {issue}"
                for issue in _optional_text_problems(slot, allow_role=True))
            # Seat names/roles are fixed product copy, not user-editable config.
            seat = COUNCIL_SEATS[index] if index < COUNCIL_SEAT_COUNT else None
            if seat is not None:
                for key, fixed in (("name", seat["name"]), ("role", seat["role"])):
                    given = slot.get(key)
                    if isinstance(given, str) and given.strip() and given.strip() != fixed:
                        problems.append(
                            f"preset '{label}' member {index + 1}: {key} is fixed to "
                            f"'{fixed}' and cannot be changed")
        assigned = [
            slot for slot in members
            if isinstance(slot, dict)
            and str(slot.get("provider") or "").strip()
            and str(slot.get("model") or "").strip()
        ]
        if not assigned:
            problems.append(f"preset '{label}': needs at least one member with a model selected")

        chairman_raw = preset.get("chairman")
        chairman_issue = slot_problem(chairman_raw)
        if chairman_issue:
            problems.append(f"preset '{label}' chairman: {chairman_issue}")
        if isinstance(chairman_raw, dict):
            problems.extend(
                f"preset '{label}' chairman: {issue}"
                for issue in _optional_text_problems(chairman_raw))

        judge = preset.get("judge")
        if isinstance(judge, dict):
            problems.extend(
                f"preset '{label}' judge: {issue}" for issue in _optional_text_problems(judge))
            provider = str(judge.get("provider") or "").strip()
            model = str(judge.get("model") or "").strip()
            if bool(provider) != bool(model):
                problems.append(
                    f"preset '{label}' judge: provider and model must both be set or both empty "
                    "(empty = members review each other)")
            elif provider:
                judge_issue = slot_problem(judge)
                if judge_issue:
                    problems.append(f"preset '{label}' judge: {judge_issue}")
    return problems


def resolve_council_preset(config: Any, name: str | None = None) -> dict[str, Any]:
    cfg = normalize_council_config(config)
    preset_name = str(name or cfg.get("default_preset") or DEFAULT_COUNCIL_PRESET_NAME).strip()
    preset = cfg["presets"].get(preset_name)
    if preset is None:
        _exc_cls = CouncilPresetNotFound
        available = ", ".join(cfg["presets"]) or "(none)"
        raise _exc_cls(
            f"Council preset '{preset_name}' was not found. Available presets: "
            f"{available}. Run `hermes council list`.")
    return deepcopy(preset)


def exact_council_preset_name(config: Any, text: str) -> str | None:
    """Return the preset name iff ``text`` exactly matches an *enabled* preset.

    Used by the no-explicit-provider switch path for a bare ``/model <preset>``; honors the
    per-preset ``enabled`` opt-out exactly like ``exact_moa_preset_name`` (#55187 semantics)."""
    wanted = str(text or "").strip()
    if not wanted:
        return None
    preset = normalize_council_config(config)["presets"].get(wanted)
    return None if preset is None or not preset.get("enabled", True) else wanted


def normalize_council_model_string(model: Any) -> tuple[str | None, Any]:
    """``council:<preset>`` -> ``("council", preset)``; anything else -> ``(None, model)``.

    Non-interactive parity with ``cli._normalize_moa_model`` (``hermes chat -Q -m
    council:<preset>``): the virtual provider is selected before provider resolution so the
    real provider never sees the preset name as a model id."""
    if isinstance(model, str) and model.strip().lower().startswith("council:"):
        preset = model.strip().split(":", 1)[1].strip()
        if preset:
            return "council", preset
    return None, model


def save_council_config(section: Any) -> dict[str, Any]:
    """Normalize *section* and persist the ``council:`` block, REPLACING any existing
    presets (never merge) so stale presets like a removed ``lite`` cannot linger after
    the collapse to a single fixed roster. Other config sections are left untouched."""
    from hermes_cli.config import load_config, save_config

    normalized = normalize_council_config(section)
    cfg = load_config()
    council_section = dict(cfg.get("council") or {}) if isinstance(cfg, dict) else {}
    # Replace the preset-bearing keys outright; keep any unrelated hand-edited keys.
    council_section.update(normalized)
    council_section["presets"] = normalized["presets"]
    save_config({"council": council_section}, merge_existing=True)
    return normalized
