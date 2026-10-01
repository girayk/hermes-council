"""Tests for hermes_cli/council_config.py — the FIXED 4-seat roster.

Seat names/roles are product constants (COUNCIL_SEATS): normalize forces every
preset onto exactly 4 positionally-aligned seats, unassigned seats stay legal,
and validate rejects extra seats or renamed/re-roled seats.
"""

from hermes_council.council_config import (
    COUNCIL_SEAT_COUNT,
    COUNCIL_SEATS,
    council_config_complete,
    normalize_council_config,
    resolve_council_preset,
    save_council_config,
    validate_council_payload,
)

_SEAT_NAMES = [seat["name"] for seat in COUNCIL_SEATS]


def _payload(members=None):
    return {
        "presets": {
            "default": {
                "members": members if members is not None else [
                    {"provider": "openrouter", "model": "m1"},
                    {"provider": "openrouter", "model": "m2"},
                ],
                "judge": {"provider": "", "model": ""},
                "chairman": {"provider": "openrouter", "model": "chair"},
            }
        }
    }


# ── normalize forces the fixed roster ────────────────────────────────────────


def test_normalize_forces_four_seats_with_fixed_names_and_roles():
    cfg = normalize_council_config(_payload())
    members = cfg["presets"]["default"]["members"]
    assert len(members) == COUNCIL_SEAT_COUNT == 4
    assert [m["name"] for m in members] == _SEAT_NAMES
    for member, seat in zip(members, COUNCIL_SEATS):
        assert member["role"] == seat["role"]


def test_normalize_overrides_user_supplied_names_and_roles():
    payload = _payload(members=[
        {"provider": "qwen", "model": "a", "name": "My Custom Name", "role": "My custom duty"},
        {"provider": "qwen", "model": "b"},
    ])
    members = normalize_council_config(payload)["presets"]["default"]["members"]
    assert members[0]["name"] == _SEAT_NAMES[0]
    assert members[0]["role"] == COUNCIL_SEATS[0]["role"]


def test_normalize_pads_missing_seats_as_unassigned():
    members = normalize_council_config(_payload())["presets"]["default"]["members"]
    assert members[0]["provider"] == "openrouter" and members[0]["enabled"] is True
    for unassigned in members[2:]:
        assert unassigned["provider"] == "" and unassigned["model"] == ""
        assert unassigned["enabled"] is False


def test_normalize_drops_extra_members_beyond_seats():
    payload = _payload(members=[
        {"provider": "qwen", "model": f"m{i}"} for i in range(6)
    ])
    members = normalize_council_config(payload)["presets"]["default"]["members"]
    assert len(members) == COUNCIL_SEAT_COUNT
    assert [m["model"] for m in members[:4]] == ["m0", "m1", "m2", "m3"]


def test_normalize_empty_members_yields_four_unassigned_seats():
    members = normalize_council_config(_payload(members=[]))["presets"]["default"]["members"]
    assert len(members) == COUNCIL_SEAT_COUNT
    assert all(m["provider"] == "" and m["enabled"] is False for m in members)


def test_normalize_drops_forbidden_slot_providers_to_unassigned():
    payload = _payload(members=[
        {"provider": "council", "model": "default"},
        {"provider": "moa", "model": "x"},
        {"provider": "qwen", "model": "ok"},
    ])
    members = normalize_council_config(payload)["presets"]["default"]["members"]
    assert members[0]["provider"] == "" and members[0]["enabled"] is False
    assert members[1]["provider"] == "" and members[1]["enabled"] is False
    assert members[2]["provider"] == "qwen" and members[2]["model"] == "ok"


def test_resolve_preset_carries_fixed_seats():
    preset = resolve_council_preset(_payload())
    assert [m["name"] for m in preset["members"]] == _SEAT_NAMES


def test_council_config_complete_semantics():
    # chairman set + >=1 assigned member
    assert council_config_complete(_payload()["presets"]["default"]) is True
    # no assigned member
    empty = {"members": [{"provider": "", "model": "", "enabled": False}],
             "chairman": {"provider": "c", "model": "d"}}
    assert council_config_complete(empty) is False
    # no chairman
    no_chair = {"members": [{"provider": "a", "model": "b"}], "chairman": {"provider": "", "model": ""}}
    assert council_config_complete(no_chair) is False


# ── validate: write-boundary rejection ──────────────────────────────────────


def test_validate_accepts_fixed_roster_payload():
    assert validate_council_payload(_payload()) == []


def test_validate_accepts_unassigned_seats():
    payload = _payload(members=[
        {"provider": "qwen", "model": "a"},
        {"provider": "", "model": ""},
        {"provider": "", "model": ""},
        {"provider": "", "model": ""},
    ])
    assert validate_council_payload(payload) == []


def test_validate_rejects_extra_seats():
    payload = _payload(members=[{"provider": "qwen", "model": f"m{i}"} for i in range(5)])
    problems = validate_council_payload(payload)
    assert any("fixed seats" in p for p in problems)


def test_validate_rejects_renamed_seat():
    payload = _payload(members=[
        {"provider": "qwen", "model": "a", "name": "Not The Skeptic"},
    ])
    problems = validate_council_payload(payload)
    assert any("name is fixed" in p for p in problems)


def test_validate_rejects_custom_role():
    payload = _payload(members=[
        {"provider": "qwen", "model": "a", "role": "Be nice to everyone."},
    ])
    problems = validate_council_payload(payload)
    assert any("role is fixed" in p for p in problems)


def test_validate_accepts_matching_fixed_name_and_role():
    seat = COUNCIL_SEATS[0]
    payload = _payload(members=[
        {"provider": "qwen", "model": "a", "name": seat["name"], "role": seat["role"]},
    ])
    assert validate_council_payload(payload) == []


def test_validate_rejects_non_string_name():
    problems = validate_council_payload(_payload(members=[
        {"provider": "qwen", "model": "a", "name": {"not": "a string"}}]))
    assert any("name" in p for p in problems)


def test_validate_rejects_half_filled_seat():
    problems = validate_council_payload(_payload(members=[{"provider": "qwen", "model": ""}]))
    assert problems


def test_validate_rejects_missing_chairman():
    payload = _payload()
    payload["presets"]["default"]["chairman"] = {"provider": "", "model": ""}
    problems = validate_council_payload(payload)
    assert any("chairman" in p for p in problems)


# ── save/load round-trip ────────────────────────────────────────────────────


def test_save_and_load_preserves_fixed_roster(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = save_council_config(_payload())
    assert [m["name"] for m in saved["presets"]["default"]["members"]] == _SEAT_NAMES

    from hermes_cli.config import load_config
    again = normalize_council_config(load_config().get("council") or {})
    members = again["presets"]["default"]["members"]
    assert len(members) == COUNCIL_SEAT_COUNT
    assert members[0]["provider"] == "openrouter" and members[0]["model"] == "m1"
    assert members[0]["role"] == COUNCIL_SEATS[0]["role"]
