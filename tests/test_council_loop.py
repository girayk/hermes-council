"""Tests for agent/council_loop.py — the LLM Council fork of the MoA runtime.

All LLM traffic is mocked at ``agent.auxiliary_client.call_llm`` (the single
delegation point of ``council_loop._call_llm``); no live calls are made.

Coverage:
  (a) 3 members, no judge  -> Stage 2 runs 3 ``council_review`` calls, the
      FINAL RANKING block is parsed and the guidance carries the digest + header
  (b) judge set            -> exactly one ``council_judge`` call, zero reviews
  (c) lite preset          -> no Stage 2 at all
  (d) a member that raises -> dropped with a failure note; others still produce guidance
  (e) chairman call        -> receives the real tools and stream/stream_options passthrough
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


def _write_config(home, *, lite=False, judge=None, members=3, fail_member=None):
    judge = judge or {"provider": "", "model": ""}
    member_lines = "\n".join(
        f"        - provider: openrouter\n          model: member-{i}\n          enabled: true"
        for i in range(1, members + 1)
    )
    (home / "config.yaml").write_text(
        "council:\n"
        "  default_preset: default\n"
        "  presets:\n"
        "    default:\n"
        "      members:\n"
        f"{member_lines}\n"
        "      judge:\n"
        f"        provider: \"{judge.get('provider', '')}\"\n"
        f"        model: \"{judge.get('model', '')}\"\n"
        "        enabled: true\n"
        "      chairman:\n"
        "        provider: openrouter\n"
        "        model: chairman-model\n"
        f"      lite: {'true' if lite else 'false'}\n"
        "      enabled: true\n",
        encoding="utf-8",
    )


@pytest.fixture
def council_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _install_fake(monkeypatch, ranking="FINAL RANKING:\n1. Response A\n2. Response B\n3. Response C",
                  fail_models=()):
    """Mock call_llm at the auxiliary_client module (council_loop._call_llm delegates there)."""
    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        task = kwargs.get("task")
        if task == "council_member":
            model = str(kwargs.get("model") or "")
            if model in fail_models:
                raise RuntimeError("member exploded")
            return _response(f"answer from slot {len(calls)}")
        if task in ("council_review", "council_judge"):
            return _response(ranking)
        return _response("chairman acted")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    return calls


def _facade():
    from hermes_council.council_loop import CouncilChatCompletions
    return CouncilChatCompletions("default")


# ---------------------------------------------------------------------------
# (a) peer review: 3 members, no judge
# ---------------------------------------------------------------------------

def test_peer_review_three_members_ranks_and_guidance(council_home, monkeypatch):
    _write_config(council_home, members=3)
    calls = _install_fake(monkeypatch)
    facade = _facade()

    prepared = facade.prepare(messages=[{"role": "user", "content": "plan the migration"}])

    tasks = [c["task"] for c in calls]
    assert tasks.count("council_member") == 3
    assert tasks.count("council_review") == 3
    assert tasks.count("council_judge") == 0
    # Member/review calls never receive tools.
    assert all("tools" not in c or not c.get("tools") for c in calls if c["task"] != "council_chairman")

    guidance = prepared["guidance"]
    assert "[LLM Council" in guidance
    assert "Preset: default" in guidance
    assert "Chairman/acting model: openrouter:chairman-model" in guidance
    assert "FINAL RANKING digest" in guidance
    # Digest carries avg rank + votes for every member answer.
    assert "avg 1.00 (3 votes)" in guidance
    # De-anonymized member answers carry the FIXED seat names.
    assert "Response A (First-Principles Analyst (openrouter:member-1))" in guidance
    assert "Response C (Pragmatist (openrouter:member-3))" in guidance
    assert "Reviewer First-Principles Analyst (openrouter:member-1)" in guidance
    # Reviewer excerpts with their ranking.
    assert "1. Response A" in guidance
    assert "chairman of an LLM council" in guidance


def test_ranking_parse_strict_and_fallback():
    from hermes_council.council_loop import aggregate_rankings, parse_ranking

    strict = parse_ranking(
        "blah blah\nFINAL RANKING:\n1. Response B\n2. Response A\n3. Response C"
    )
    assert strict == ["Response B", "Response A", "Response C"]

    # Fallback: no marker, bare label order.
    fallback = parse_ranking("I liked Response C best, then Response A, then Response B.")
    assert fallback == ["Response C", "Response A", "Response B"]

    # No parseable ranking at all.
    assert parse_ranking("no labels here") == []

    digest = aggregate_rankings(
        [["Response A", "Response B"], ["Response B", "Response A"]],
        ["Response A", "Response B"],
    )
    # Tie on avg rank (1.5 each) -> stable label order; one first-place vote each.
    assert digest == [("Response A", 1.5, 1), ("Response B", 1.5, 1)]


# ---------------------------------------------------------------------------
# (b) judge set: exactly one council_judge call, zero council_review
# ---------------------------------------------------------------------------

def test_judge_slot_replaces_peer_review(council_home, monkeypatch):
    _write_config(council_home, members=3, judge={"provider": "openrouter", "model": "judge-model"})
    calls = _install_fake(monkeypatch)
    facade = _facade()

    prepared = facade.prepare(messages=[{"role": "user", "content": "plan the migration"}])

    tasks = [c["task"] for c in calls]
    assert tasks.count("council_member") == 3
    assert tasks.count("council_judge") == 1
    assert tasks.count("council_review") == 0
    judge_call = next(c for c in calls if c["task"] == "council_judge")
    assert judge_call["model"] == "judge-model"
    assert "FINAL RANKING digest" in prepared["guidance"]
    # Anonymized review prompt must not leak member model ids in the response headers.
    review_text = "\n".join(
        str(m.get("content")) for m in judge_call["messages"]
    )
    assert "openrouter:member-1" not in review_text
    assert "Response A" in review_text
    assert "FINAL RANKING:" in review_text  # strict ranking instructions included


# ---------------------------------------------------------------------------
# (c) lite: no Stage 2 at all
# ---------------------------------------------------------------------------

def test_lite_preset_skips_stage2(council_home, monkeypatch):
    _write_config(council_home, members=3, lite=True)
    calls = _install_fake(monkeypatch)
    facade = _facade()

    prepared = facade.prepare(messages=[{"role": "user", "content": "plan the migration"}])

    tasks = [c["task"] for c in calls]
    assert tasks.count("council_member") == 3
    assert "council_review" not in tasks
    assert "council_judge" not in tasks
    assert "FINAL RANKING digest" not in (prepared["guidance"] or "")
    assert "[LLM Council" in prepared["guidance"]


def test_single_answer_skips_stage2(council_home, monkeypatch):
    _write_config(council_home, members=1)
    calls = _install_fake(monkeypatch)
    facade = _facade()

    facade.prepare(messages=[{"role": "user", "content": "hello"}])

    tasks = [c["task"] for c in calls]
    assert tasks.count("council_member") == 1
    assert "council_review" not in tasks
    assert "council_judge" not in tasks


# ---------------------------------------------------------------------------
# (d) failing member is dropped; others still produce guidance
# ---------------------------------------------------------------------------

def test_failed_member_dropped_with_note(council_home, monkeypatch):
    # member-2 raises inside the fake call_llm.
    _write_config(council_home, members=3)
    calls = _install_fake(monkeypatch, fail_models=("member-2",),
                          ranking="FINAL RANKING:\n1. Response A\n2. Response B")
    facade = _facade()

    prepared = facade.prepare(messages=[{"role": "user", "content": "plan the migration"}])

    guidance = prepared["guidance"]
    assert guidance
    assert "[LLM Council" in guidance
    # Failure notes carry the fixed seat name + model id.
    assert "Skeptic (openrouter:member-2) [failed:" in guidance
    # Surviving members still answered (labels A and B, re-lettered past the drop).
    assert "Response A (First-Principles Analyst (openrouter:member-1))" in guidance
    assert "Response B (Pragmatist (openrouter:member-3))" in guidance
    tasks = [c["task"] for c in calls]
    assert tasks.count("council_review") == 2  # only the two survivors review


def test_all_members_failed_still_yields_guidance(council_home, monkeypatch):
    _write_config(council_home, members=3)

    def fake_call_llm(**kwargs):
        if kwargs.get("task") == "council_member":
            raise RuntimeError("total outage")
        return _response("chairman acted")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    facade = _facade()

    prepared = facade.prepare(messages=[{"role": "user", "content": "hello"}])

    guidance = prepared["guidance"]
    assert guidance
    assert "All council members failed" in guidance
    assert "[failed:" in guidance


# ---------------------------------------------------------------------------
# (e) chairman receives tools + stream passthrough
# ---------------------------------------------------------------------------

def test_chairman_gets_tools_and_stream_passthrough(council_home, monkeypatch):
    _write_config(council_home, members=2, lite=True)
    calls = _install_fake(monkeypatch)
    facade = _facade()

    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
    result = facade.create(
        model="default",
        messages=[{"role": "user", "content": "run it"}],
        tools=tools,
        stream=True,
        max_tokens=1234,
        temperature=0.25,
    )

    chair_calls = [c for c in calls if c["task"] == "council_chairman"]
    assert len(chair_calls) == 1
    chair = chair_calls[0]
    assert chair["model"] == "chairman-model"
    assert chair["provider"] == "openrouter"
    assert chair["tools"] is not None and len(chair["tools"]) >= 1
    assert chair["stream"] is True
    assert chair["stream_options"] == {"include_usage": True}
    assert chair["max_tokens"] == 1234
    # Guidance attached as its own trailing user message on the chairman request.
    assert chair["messages"][-1]["role"] == "user"
    assert "[LLM Council" in str(chair["messages"][-1]["content"])
    # stream=True with a completed response -> one-chunk stream adapter (MoA parity).
    chunks = list(result)
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "chairman acted"

    # Non-streaming chairman call returns the completed response unchanged.
    calls.clear()
    response = facade.create(
        model="default",
        messages=[{"role": "user", "content": "second turn, new request"}],
        tools=tools,
        stream=False,
    )
    chair = next(c for c in calls if c["task"] == "council_chairman")
    assert "stream" not in chair
    assert response.choices[0].message.content == "chairman acted"


# ---------------------------------------------------------------------------
# fan-out caching + facade identity
# ---------------------------------------------------------------------------

def test_fanout_cache_reused_within_user_turn(council_home, monkeypatch):
    _write_config(council_home, members=2, lite=True)
    calls = _install_fake(monkeypatch)
    facade = _facade()

    base = [{"role": "user", "content": "do the task"}]
    facade.create(model="default", messages=base, tools=[], stream=False)
    member_calls_first = sum(1 for c in calls if c["task"] == "council_member")

    # Later tool iteration of the SAME user turn: guidance cached, only chairman re-called.
    grown = [*base,
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                                  "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"}]
    calls.clear()
    facade.create(model="default", messages=grown, tools=[], stream=False)
    assert sum(1 for c in calls if c["task"] == "council_member") == 0
    assert sum(1 for c in calls if c["task"] == "council_chairman") == 1
    assert member_calls_first == 2


def test_bind_council_runtime_pins_virtual_identity():
    from hermes_council.council_loop import bind_council_runtime

    agent = SimpleNamespace()
    bind_council_runtime(agent, "default")

    assert agent.model == "default"
    assert agent.provider == "council"
    assert agent.requested_provider == "council"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "council-virtual-provider"
    assert agent.base_url == "council://local"
    assert agent._client_kwargs == {}
    from hermes_council.council_loop import CouncilClient
    assert isinstance(agent.client, CouncilClient)
    assert agent.client.last_chairman_slot is None


def test_consume_member_usage_shape(council_home, monkeypatch):
    _write_config(council_home, members=2, lite=True)
    _install_fake(monkeypatch)
    from hermes_council.council_loop import CouncilClient
    client = CouncilClient("default")
    client.chat.completions.create(
        model="default", messages=[{"role": "user", "content": "go"}], stream=False,
    )
    usage, cost = client.consume_member_usage()
    assert usage is not None
    # Reset after consumption (double-count guard).
    usage2, cost2 = client.consume_member_usage()
    assert cost2 is None
    assert client.last_chairman_slot == {"provider": "openrouter", "model": "chairman-model"}


# ---------------------------------------------------------------------------
# (f) FIXED SEATS: stage-1 role system message, seat-name labels, stage-2 anonymity
# ---------------------------------------------------------------------------

from hermes_council.council_config import COUNCIL_SEATS  # noqa: E402

_SEAT1_NAME, _SEAT1_ROLE = COUNCIL_SEATS[0]["name"], COUNCIL_SEATS[0]["role"]
_SEAT2_NAME, _SEAT2_ROLE = COUNCIL_SEATS[1]["name"], COUNCIL_SEATS[1]["role"]


def _write_seat_config(home, *, chairman_name="", judge=None, members=None):
    """Config with 3 assigned seats. Any name/role the raw config carries must be
    IGNORED by normalize (seats are fixed); we include bogus ones to prove it."""
    judge = judge or {"provider": "", "model": ""}
    members = members if members is not None else [
        {"model": "member-1", "name": "BogusName", "role": "Bogus duty"},
        {"model": "member-2"},
        {"model": "member-3"},
    ]
    member_lines = []
    for member in members:
        block = (f"        - provider: openrouter\n"
                 f"          model: {member['model']}\n"
                 f"          enabled: true")
        if member.get("name"):
            block += f"\n          name: \"{member['name']}\""
        if member.get("role"):
            block += f"\n          role: \"{member['role']}\""
        member_lines.append(block)
    chairman_block = ("      chairman:\n"
                      "        provider: openrouter\n"
                      "        model: chairman-model")
    if chairman_name:
        chairman_block += f"\n        name: \"{chairman_name}\""
    (home / "config.yaml").write_text(
        "council:\n"
        "  default_preset: default\n"
        "  presets:\n"
        "    default:\n"
        "      members:\n"
        + "\n".join(member_lines) + "\n"
        "      judge:\n"
        f"        provider: \"{judge.get('provider', '')}\"\n"
        f"        model: \"{judge.get('model', '')}\"\n"
        "        enabled: true\n"
        + chairman_block + "\n"
        "      lite: false\n"
        "      enabled: true\n",
        encoding="utf-8",
    )


def _member_calls_by_model(calls, task="council_member"):
    return {str(c.get("model")): c for c in calls if c.get("task") == task}


def _all_message_text(call):
    return "\n".join(str(m.get("content")) for m in call.get("messages") or [])


def test_assigned_seat_gets_fixed_role_system_message(council_home, monkeypatch):
    _write_seat_config(council_home)
    calls = _install_fake(monkeypatch)
    _facade().prepare(messages=[{"role": "user", "content": "plan the migration"}])

    by_model = _member_calls_by_model(calls)
    # Seat 1 (First-Principles Analyst): exactly one extra system message carrying
    # the FIXED seat name and role — not the bogus config values.
    msgs = by_model["member-1"]["messages"]
    role_msgs = [m for m in msgs if m["role"] == "system" and _SEAT1_NAME in str(m["content"])]
    assert len(role_msgs) == 1
    text = str(role_msgs[0]["content"])
    assert text.startswith(f"You are {_SEAT1_NAME}, a member of an LLM council.")
    assert f"Your duty on this council: {_SEAT1_ROLE}." in text
    assert "BogusName" not in text and "Bogus duty" not in text

    # Seat 2 (Skeptic) gets its own fixed duty.
    msgs2 = by_model["member-2"]["messages"]
    role2 = [m for m in msgs2 if m["role"] == "system" and _SEAT2_NAME in str(m["content"])]
    assert len(role2) == 1
    assert f"Your duty on this council: {_SEAT2_ROLE}." in str(role2[0]["content"])

    # Each member sees ONLY its own duty, never another seat's role text.
    assert _SEAT2_ROLE not in _all_message_text(by_model["member-1"])
    assert _SEAT1_ROLE not in _all_message_text(by_model["member-2"])


def test_guidance_and_events_use_seat_names(council_home, monkeypatch):
    _write_seat_config(council_home, chairman_name="ChairName")
    _install_fake(monkeypatch)
    events: list[tuple] = []
    from hermes_council.council_loop import CouncilChatCompletions
    facade = CouncilChatCompletions(
        "default", reference_callback=lambda event, **kw: events.append((event, kw)))
    prepared = facade.prepare(messages=[{"role": "user", "content": "plan the migration"}])

    guidance = prepared["guidance"]
    # De-anonymized member answers carry the fixed seat names; bogus config names never appear.
    assert f"Response A ({_SEAT1_NAME} (openrouter:member-1))" in guidance
    assert f"Response B ({_SEAT2_NAME} (openrouter:member-2))" in guidance
    assert "BogusName" not in guidance
    assert f"Reviewer {_SEAT1_NAME} (openrouter:member-1)" in guidance
    assert "Chairman/acting model: ChairName (openrouter:chairman-model)" in guidance

    by_event: dict[str, list] = {}
    for event, kw in events:
        by_event.setdefault(event, []).append(kw)
    # council.member payload carries the seat name.
    member_by_label = {kw["label"]: kw for kw in by_event.get("council.member", [])}
    assert member_by_label[f"{_SEAT1_NAME} (openrouter:member-1)"]["name"] == _SEAT1_NAME
    # council.review reviewer label + phase chairman label carry names.
    reviewers = {kw["reviewer"] for kw in by_event.get("council.review", [])}
    assert f"{_SEAT1_NAME} (openrouter:member-1)" in reviewers
    chairmen = {kw["chairman"] for kw in by_event.get("council.phase", [])}
    assert "ChairName (openrouter:chairman-model)" in chairmen


def test_stage2_anonymity_peer_review_hides_names_and_roles(council_home, monkeypatch):
    """karpathy anti-favoritism rule: reviewers see ONLY 'Response A/B/C' labels —
    never a seat name, role string, or model id."""
    _write_seat_config(council_home)
    calls = _install_fake(monkeypatch)
    _facade().prepare(messages=[{"role": "user", "content": "plan the migration"}])

    reviews = [c for c in calls if c["task"] in ("council_review", "council_judge")]
    assert len(reviews) == 3
    for call in reviews:
        blob = _all_message_text(call)
        for seat in COUNCIL_SEATS:
            assert seat["name"] not in blob
            assert seat["role"] not in blob
        assert "BogusName" not in blob
        assert "openrouter:member-1" not in blob
        assert "Response A" in blob


def test_stage2_anonymity_judge_hides_names_and_roles(council_home, monkeypatch):
    _write_seat_config(
        council_home, chairman_name="ChairName",
        judge={"provider": "openrouter", "model": "judge-model"})
    calls = _install_fake(monkeypatch)
    _facade().prepare(messages=[{"role": "user", "content": "plan the migration"}])

    judge_calls = [c for c in calls if c["task"] == "council_judge"]
    assert len(judge_calls) == 1
    blob = _all_message_text(judge_calls[0])
    for seat in COUNCIL_SEATS:
        assert seat["name"] not in blob
        assert seat["role"] not in blob
    assert "ChairName" not in blob
    assert "Response A" in blob


def test_failed_seat_uses_seat_name_label(council_home, monkeypatch):
    _write_seat_config(council_home, members=[
        {"model": "member-1"},
        {"model": "member-2"},
    ])
    _install_fake(monkeypatch, fail_models=("member-1",),
                  ranking="FINAL RANKING:\n1. Response A")
    prepared = _facade().prepare(messages=[{"role": "user", "content": "plan the migration"}])
    guidance = prepared["guidance"]
    assert f"{_SEAT1_NAME} (openrouter:member-1) [failed:" in guidance
    assert f"Response A ({_SEAT2_NAME} (openrouter:member-2))" in guidance


def test_unassigned_seats_are_skipped(council_home, monkeypatch):
    """Seats without a model are never called (only assigned ones fan out)."""
    _write_seat_config(council_home, members=[{"model": "member-1"}])  # 1 assigned, 3 unassigned
    calls = _install_fake(monkeypatch, ranking="FINAL RANKING:\n1. Response A")
    from hermes_council.council_loop import CouncilChatCompletions
    facade = CouncilChatCompletions("default")
    resp = facade.create(model="default",
                         messages=[{"role": "user", "content": "plan the migration"}])
    assert resp is not None
    tasks = [c["task"] for c in calls]
    assert tasks.count("council_member") == 1
    # Single answer => stage 2 skipped (peer review needs >=2 answers).
    assert tasks.count("council_review") == 0
    assert tasks.count("council_chairman") == 1
