"""LLM Council runtime helpers — slim fork of agent/moa_loop.py (karpathy/llm-council semantics).

Council is a virtual provider (slug ``council``) like MoA: named presets appear in the
model menu; selecting one makes the preset's *chairman* the acting model (tools +
streaming) after a 3-stage loop:

  Stage 1  enabled members answer the same advisory view in parallel (``council_member``)
  Stage 2  members review + rank each other anonymously (``council_review``), or a
           dedicated judge ranks them in one call (``council_judge``); skipped when the
           preset is ``lite`` or fewer than two members answered
  Stage 3  the chairman acts as the model (``council_chairman``) with the real tools

Slim by contract: no trace persistence, no privacy filter, no fan-out cadence options,
no reasoning_effort UI. Advisory-view construction, guidance attachment, slot runtime
resolution, accounting structs and the alternation-recovery helpers are IMPORTED from
agent.moa_loop / agent.moa_alternation — never copied — so MoA and Council cannot drift.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
from typing import Any

from agent import auxiliary_client as _auxiliary_client
from agent.message_content import flatten_message_text  # noqa: F401  (re-export parity with moa_loop)
from agent.moa_alternation import destination_key, is_role_alternation_rejection, merge_same_role_messages
from agent.moa_loop import (
    _ADVISORY_INSTRUCTION,
    _RefAccounting,
    _STALE_GUIDANCE_NOTE,
    _agent_cache_opts,
    _attach_reference_guidance,
    _completed_response_as_stream_chunk,
    _extract_text,
    _hash_messages,
    _maybe_apply_moa_cache_control,
    _merge_slot_extra_body,
    _price_reference_response,
    _reference_messages,
    _slot_label,
    _slot_reasoning_config,
    _slot_runtime,
    _sum_reference_accounting,
    _tool_activity_since_last_user,
    _trim_messages_for_reference,
    _with_cache_disabled,
    peel_reference_guidance,
)
from agent.usage_pricing import CanonicalUsage

logger = logging.getLogger(__name__)

__all__ = [
    "CouncilChatCompletions",
    "CouncilClient",
    "build_council_facade",
    "bind_council_runtime",
    "resolve_council_chairman",
    "peel_reference_guidance",
]


def _call_llm(**kwargs: Any) -> Any:
    """Indirection so tests (and the aux layer) see one call site: agent.auxiliary_client.call_llm."""
    return _auxiliary_client.call_llm(**kwargs)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap on concurrent member/review calls (guards pathologically large presets).
_MAX_COUNCIL_WORKERS = 8

# Hard wall-clock deadline per stage so a wedged member never hangs the turn
# (mirrors auxiliary.moa_reference.timeout's 900s default). Presets may override
# with ``stage_timeout`` seconds.
_COUNCIL_STAGE_TIMEOUT_S = 900.0
_COUNCIL_POLL_INTERVAL_S = 5.0

# Review-text clip for the guidance block.
_REVIEW_CLIP_CHARS = 4000

# karpathy/llm-council RANKING_INSTRUCTIONS (backend/council.py) — reproduced verbatim
# per the shared contract so the strict FINAL RANKING block parses.
RANKING_INSTRUCTIONS = (
    "IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:\n"
    '- Start with the line "FINAL RANKING:" (all caps, with colon)\n'
    "- Then list the responses from best to worst as a numbered list\n"
    '- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")\n'
    "- Do not add any other text or explanations in the ranking section"
)

_MEMBER_SYSTEM_PROMPT = (
    "You are a member of an LLM council. The council members each answer the same "
    "request independently; a chairman then weighs the answers (and the members' "
    "anonymous peer reviews) and acts. You cannot call tools, run commands, browse, "
    "or access files, and you must never claim or imply that you executed anything — "
    "only the chairman can act.\n\n"
    "Give your best, direct answer or recommendation for the conversation state "
    "below: your reasoning, your concrete recommendation, and any risks or "
    "disagreements you anticipate. Respond in prose — never emit a tool call or a "
    "JSON tool-call object."
)

_REVIEWER_SYSTEM_PROMPT = (
    "You are an impartial reviewer in an LLM council. You will see the conversation "
    "and several anonymized candidate responses from the other council members. "
    "Evaluate each response for correctness, completeness and usefulness, then rank "
    "them from best to worst. You cannot call tools and must never claim to have "
    "executed anything. Respond in prose, ending with the strict ranking block."
)

_CHAIRMAN_INSTRUCTION = (
    "You are the chairman of an LLM council. The members above answered the request "
    "independently and reviewed each other anonymously (or a dedicated judge ranked "
    "them). Weigh the peer ranking and the member answers, reconcile disagreements, "
    "and then answer the user directly or call tools as needed — you are the acting "
    "model and the only one that can act."
)

# Stage-1 duty prompt for a member slot with a configured ``role``. Injected as an
# extra leading system message on THAT member's call only — it never enters the shared
# member view and never reaches stage 2 (reviewers/judge see anonymous 'Response X'
# labels only: karpathy's anti-favoritism rule).
_MEMBER_ROLE_PROMPT = (
    "You are {name}, a member of an LLM council. Your duty on this council: {role}. "
    "Answer the user's question from that perspective, as rigorously as you can."
)


# ---------------------------------------------------------------------------
# Preset resolution (cached per config-file signature, like moa_loop)
# ---------------------------------------------------------------------------

_preset_cache_lock = threading.Lock()
_preset_cache: dict[tuple, Any] = {}


def _normalize_council_presets(council_raw: Any) -> dict[str, dict[str, Any]]:
    """Local fallback normalizer for the ``council:`` config block (contract §1)."""
    raw = council_raw if isinstance(council_raw, dict) else {}
    presets_raw = raw.get("presets") if isinstance(raw.get("presets"), dict) else {}
    presets: dict[str, dict[str, Any]] = {}
    for name, entry in presets_raw.items():
        preset = entry if isinstance(entry, dict) else {}
        members = [m for m in (preset.get("members") or []) if isinstance(m, dict)]
        judge = preset.get("judge") if isinstance(preset.get("judge"), dict) else {}
        chairman = preset.get("chairman") if isinstance(preset.get("chairman"), dict) else {}
        presets[str(name)] = {
            "members": members,
            "judge": judge,
            "chairman": chairman,
            "lite": bool(preset.get("lite")),
            "enabled": bool(preset.get("enabled", True)),
            "member_temperature": preset.get("member_temperature"),
            "chairman_temperature": preset.get("chairman_temperature"),
            "stage_timeout": preset.get("stage_timeout"),
        }
    return presets


from .council_config import CouncilPresetNotFound as CouncilPresetNotFoundError


def _resolve_council_preset(council_raw: Any, name: str | None = None) -> dict[str, Any]:
    """Resolve one council preset; prefers the sibling council_config module."""
    try:
        from .council_config import resolve_council_preset
    except ImportError:
        resolve_council_preset = None
    if resolve_council_preset is not None:
        try:
            return resolve_council_preset(council_raw, name)
        except CouncilPresetNotFoundError:
            raise
        except Exception as exc:
            # A sibling-module resolution bug must not wedge the turn; fall through
            # to the local normalizer (same contract shape).
            logger.debug("council_config.resolve_council_preset failed (%s); using local resolver", exc)
    cfg = council_raw if isinstance(council_raw, dict) else {}
    preset_name = str(name or cfg.get("default_preset") or "default").strip()
    preset = _normalize_council_presets(cfg).get(preset_name)
    if preset is None:
        available = ", ".join(_normalize_council_presets(cfg)) or "(none)"
        raise CouncilPresetNotFoundError(
            f"Council preset '{preset_name}' was not found. Available presets: "
            f"{available}. Run `hermes council list`.")
    return preset


def _resolve_preset_cached(preset_name: str) -> dict[str, Any]:
    """Resolved council preset, cached per config-file signature (mirrors moa_loop)."""
    from hermes_cli.config import get_config_path, load_config
    from utils import file_signature
    try:
        cfg_stamp = file_signature(get_config_path().stat())
    except OSError:
        cfg_stamp = None
    council_raw = load_config().get("council") or {}
    key = (cfg_stamp, preset_name)
    with _preset_cache_lock:
        preset = _preset_cache.get(key) if cfg_stamp is not None else None
    if preset is None:
        preset = _resolve_council_preset(council_raw, preset_name)
        if cfg_stamp is not None:
            with _preset_cache_lock:
                _preset_cache.clear()  # one live config stamp at a time
                _preset_cache[key] = preset
    return preset


def resolve_council_chairman(preset_name: str | None) -> tuple[str | None, str | None]:
    """Council preset → chairman (provider, model); (None, None) if unresolvable.

    Shared by auxiliary_client's virtual-provider unwraps so the lookup cannot drift
    (mirrors ``_resolve_moa_aggregator``).
    """
    try:
        preset = _resolve_preset_cached(str(preset_name or "default"))
        chairman = preset.get("chairman") or {}
        provider = str(chairman.get("provider") or "").strip()
        model = str(chairman.get("model") or "").strip()
        if provider and model and provider.lower() not in ("council", "moa"):
            return provider, model
    except Exception:
        logger.debug("Council chairman resolution failed for preset %r", preset_name, exc_info=True)
    return None, None


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------

def _enabled_members(preset: dict[str, Any]) -> list[dict[str, Any]]:
    """Enabled member slots with a complete (provider, model); council slots are skipped."""
    out = []
    for slot in preset.get("members") or []:
        if not slot.get("enabled", True):
            continue
        if not str(slot.get("provider") or "").strip() or not str(slot.get("model") or "").strip():
            continue  # unset slot
        if str(slot.get("provider") or "").strip().lower() == "council":
            continue  # recursion guard (config validation also rejects it)
        out.append(slot)
    return out


def _judge_slot(preset: dict[str, Any]) -> dict[str, Any] | None:
    """The judge slot when SET (non-empty provider AND model, enabled); else None (peer review)."""
    judge = preset.get("judge") or {}
    if not judge.get("enabled", True):
        return None
    provider = str(judge.get("provider") or "").strip()
    model = str(judge.get("model") or "").strip()
    if provider and model and provider.lower() != "council":
        return judge
    return None


def _preset_temperature(preset: dict[str, Any], key: str) -> float | None:
    """Optional preset temperature; None (absent/empty/null) = provider default."""
    value = preset.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("ignoring non-numeric %s=%r in Council preset", key, value)
        return None


def _stage_timeout(preset: dict[str, Any]) -> float:
    raw = preset.get("stage_timeout")
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return _COUNCIL_STAGE_TIMEOUT_S


def _label_letter(index: int) -> str:
    """Anonymized response label suffix: A, B, … Z, then 27, 28, … (pathological presets)."""
    return chr(ord("A") + index) if index < 26 else str(index + 1)

def _slot_name(slot: dict[str, Any]) -> str:
    """Optional display name for a council slot ("" when unset)."""
    return str(slot.get("name") or "").strip()

def _council_slot_label(slot: dict[str, Any]) -> str:
    """Display label: ``NAME (provider:model)`` when the slot has a name, else the plain
    MoA-style ``provider:model`` label. Used everywhere the slot is shown DE-anonymized
    (guidance block, events, failure notes) — never in stage-2 prompts."""
    name = _slot_name(slot)
    base = _slot_label(slot)
    return f"{name} ({base})" if name else base

def _member_role_message(slot: dict[str, Any]) -> dict[str, Any] | None:
    """Stage-1 duty system message for a member slot with a non-empty ``role``; None when
    the slot has no role (its messages then stay exactly as before). The name falls back
    to the plain provider:model id."""
    role = str(slot.get("role") or "").strip()
    if not role:
        return None
    name = _slot_name(slot) or f"{str(slot.get('provider') or '').strip()}:{str(slot.get('model') or '').strip()}"
    return {"role": "system", "content": _MEMBER_ROLE_PROMPT.format(name=name, role=role)}


def _clip(text: str, budget: int = _REVIEW_CLIP_CHARS) -> str:
    if not text or len(text) <= budget:
        return text
    return text[:budget] + f"\n[... {len(text) - budget} chars clipped ...]"


# ---------------------------------------------------------------------------
# Stage 2: ranking parse + aggregation
# ---------------------------------------------------------------------------

_RANK_LINE_RE = re.compile(r"^\s*\d+\.\s*Response\s+([A-Z]+|\d+)\s*$", re.IGNORECASE)
_BARE_LABEL_RE = re.compile(r"Response\s+([A-Z]+|\d+)", re.IGNORECASE)


def parse_ranking(text: Any) -> list[str]:
    """Parse a reviewer's ``FINAL RANKING:`` block into ordered ``Response X`` labels.

    Strict path: numbered ``N. Response X`` lines after the LAST ``FINAL RANKING:``
    marker. Fallback: bare ``Response X`` mentions in order of first appearance.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    marker = text.upper().rfind("FINAL RANKING:")
    section = text[marker:] if marker >= 0 else text
    labels: list[str] = []
    for line in section.splitlines():
        match = _RANK_LINE_RE.match(line)
        if match:
            label = f"Response {match.group(1).upper()}"
            if label not in labels:
                labels.append(label)
    if labels:
        return labels
    # Fallback: bare "Response X" order (unique, first appearance).
    for match in _BARE_LABEL_RE.finditer(section if marker >= 0 else text):
        label = f"Response {match.group(1).upper()}"
        if label not in labels:
            labels.append(label)
    return labels


def aggregate_rankings(rankings: list[list[str]], all_labels: list[str]) -> list[tuple[str, float, int]]:
    """Aggregate per-reviewer rankings into ``(label, avg_rank, votes)`` best-first.

    A label a reviewer omitted is charged worst+1 for that reviewer (karpathy
    semantics). ``votes`` counts first-place rankings. Empty ``rankings`` (lite
    mode, unparseable reviews) yields no digest.
    """
    if not all_labels or not rankings:
        return []
    worst = len(all_labels) + 1
    sums: dict[str, list[int]] = {label: [] for label in all_labels}
    votes: dict[str, int] = {label: 0 for label in all_labels}
    for ranking in rankings:
        seen: set[str] = set()
        for position, label in enumerate(ranking, start=1):
            if label in sums and label not in seen:
                sums[label].append(position)
                seen.add(label)
        if ranking and ranking[0] in votes:
            votes[ranking[0]] += 1
    reviewers = max(1, len(rankings))
    rows = []
    for label in all_labels:
        positions = sums[label] + [worst] * (reviewers - len(sums[label]))
        rows.append((label, sum(positions) / len(positions), votes[label]))
    rows.sort(key=lambda row: (row[1], row[0]))
    return rows


# ---------------------------------------------------------------------------
# Stage workers
# ---------------------------------------------------------------------------

def _run_member(slot: dict[str, Any], member_view: list[dict[str, Any]], *, temperature: float | None,
                timeout: float | None = None, context_length_cache: Any = None,
                cache_disabled: Any = None, cache_ttl: Any = None) -> tuple[str, Any]:
    """One Stage-1 member call; returns ``(text, accounting)``. Runs in a thread pool;
    exceptions propagate to the collector (the member is dropped with a failure note)."""
    runtime = _slot_runtime(slot)
    # Named advisors: a slot with a duty/role gets an extra leading system message.
    # It lives ONLY on this member's call — never in member_view (the shared transcript)
    # and therefore never in the stage-2 review/judge prompt (anonymity guard).
    role_message = _member_role_message(slot)
    messages = [m for m in (role_message, {"role": "system", "content": _MEMBER_SYSTEM_PROMPT}) if m] + list(member_view)
    trimmed = _trim_messages_for_reference(messages, slot, runtime, context_length_cache=context_length_cache)
    trimmed = _maybe_apply_moa_cache_control(trimmed, _with_cache_disabled(runtime, cache_disabled), cache_ttl=cache_ttl)
    response = _call_llm(
        task="council_member", messages=trimmed, temperature=temperature,
        timeout=timeout, reasoning_config=_slot_reasoning_config(slot), **runtime,
    )
    text = _extract_text(response) or "(empty response)"
    usage, cost, status, source = _price_reference_response(response, slot, runtime)
    return text, _RefAccounting(usage, cost, status, source, messages=trimmed, output=text,
                                model=slot.get("model"), provider=runtime.get("provider") or slot.get("provider"),
                                temperature=temperature)


def _review_messages(member_view: list[dict[str, Any]], answers: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Reviewer/judge prompt: conversation transcript + anonymized responses + strict ranking block."""
    convo = "\n\n".join(
        f"{str(m.get('role', '')).upper()}: {flatten_message_text(m.get('content'))}" for m in member_view
    )
    rendered = "\n\n".join(f"=== {label} ===\n{text}" for label, text in answers)
    user = (
        f"Conversation under review:\n{convo}\n\n"
        f"Candidate responses (anonymized):\n\n{rendered}\n\n"
        "Evaluate each response above, then rank them from best to worst.\n\n"
        f"{RANKING_INSTRUCTIONS}"
    )
    return [{"role": "system", "content": _REVIEWER_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _run_reviewer(slot: dict[str, Any], task: str, review_messages: list[dict[str, Any]], *,
                  timeout: float | None = None, context_length_cache: Any = None,
                  cache_disabled: Any = None, cache_ttl: Any = None) -> tuple[str, Any]:
    """One Stage-2 review/judge call; returns ``(text, accounting)``. Raises on failure."""
    runtime = _slot_runtime(slot)
    trimmed = _trim_messages_for_reference(review_messages, slot, runtime, context_length_cache=context_length_cache)
    trimmed = _maybe_apply_moa_cache_control(trimmed, _with_cache_disabled(runtime, cache_disabled), cache_ttl=cache_ttl)
    response = _call_llm(
        task=task, messages=trimmed, temperature=None, timeout=timeout,
        reasoning_config=_slot_reasoning_config(slot), **runtime,
    )
    text = _extract_text(response) or "(empty response)"
    usage, cost, status, source = _price_reference_response(response, slot, runtime)
    return text, _RefAccounting(usage, cost, status, source, messages=trimmed, output=text,
                                model=slot.get("model"), provider=runtime.get("provider") or slot.get("provider"),
                                temperature=None)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

_STAGE_DEADLINE_NOTE = "[failed: stage deadline exceeded]"
_INTERRUPTED_NOTE = "[skipped: interrupted by user]"


class CouncilChatCompletions:
    """OpenAI-chat-compatible facade where the chairman is the acting model.

    ``reference_callback(event, **kwargs)`` is the optional display hook (events:
    ``council.member``, ``council.review``, ``council.phase``, ``council.synthesizing``).
    ``agent`` is the owning AIAgent (lets the stage waits observe ``_interrupt_requested``).
    """

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.preset_name = preset_name or "default"
        self.reference_callback = reference_callback
        self._agent = agent
        # Turn-scoped council cache (user_turn fanout semantics, like MoA's default).
        self._council_cache_key: tuple | None = None
        self._council_cache_state: dict[str, Any] | None = None
        # Member/review/judge spend awaiting consume_member_usage().
        self._pending_member_usage: Any = CanonicalUsage()
        self._pending_member_cost: Any = None
        self._accounting_lock = threading.Lock()
        # Real chairman slot so cost accounting prices the acting turn at its model.
        self.last_chairman_slot: Any = None
        # Destinations that 400'd on adjacent same-role messages (agent/moa_alternation.py).
        self._merge_same_role_destinations: set[tuple[str, str]] = set()

    # -- accounting ---------------------------------------------------------

    def consume_member_usage(self) -> tuple[Any, Any]:
        """Pop pending fan-out ``(CanonicalUsage, cost_usd_or_None)`` and reset both."""
        with self._accounting_lock:
            usage = self._pending_member_usage or CanonicalUsage()
            cost = self._pending_member_cost
            self._pending_member_usage = CanonicalUsage()
            self._pending_member_cost = None
        return usage, cost

    def _fold_pending_accounting(self, usage: Any, cost: Any) -> None:
        with self._accounting_lock:
            self._pending_member_usage = (self._pending_member_usage or CanonicalUsage()) + usage
            if cost is not None:
                self._pending_member_cost = (self._pending_member_cost or 0) + cost

    def _fold_accountings(self, accountings: list[Any]) -> None:
        tuples = [("", "", acct) for acct in accountings if isinstance(acct, _RefAccounting)]
        if tuples:
            self._fold_pending_accounting(*_sum_reference_accounting(tuples))

    # -- display --------------------------------------------------------------

    def _emit(self, event: str, **kwargs: Any) -> None:
        if self.reference_callback is None:
            return
        try:
            self.reference_callback(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("Council reference_callback failed for %s: %s", event, exc)

    # -- prepared-request surface (mirrors MoAChatCompletions) ----------------

    def prepare(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run Stages 1-2 and return the exact chairman request (loop measures it, then
        hands it back to ``create()`` via ``_moa_prepared_request``)."""
        return self.create(messages=messages, _moa_prepare_only=True)

    def rebase_prepared_request(self, prepared: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Re-attach already-generated guidance to a rebuilt (compressed) transcript."""
        guidance = prepared.get("guidance")
        chairman_messages = [dict(message) for message in messages]
        if guidance:
            _attach_reference_guidance(chairman_messages, str(guidance))
        return {**prepared, "messages": chairman_messages}

    # -- stages ----------------------------------------------------------------

    def _run_stage(self, jobs: list[tuple[Any, tuple, dict]], *, timeout: float) -> list[Any]:
        """Run stage jobs in parallel with a HARD wall-clock deadline.

        Returns one entry per job: the worker's return value, the raised exception, or
        the deadline/interrupt note string. A wedged member never hangs the turn.
        """
        results: list[Any] = [None] * len(jobs)
        if not jobs:
            return results
        from tools.thread_context import propagate_context_to_thread
        executor = ThreadPoolExecutor(max_workers=min(_MAX_COUNCIL_WORKERS, len(jobs)))
        futures: dict[Any, int] = {}
        pending: set[Any] = set()
        interrupted = False
        try:
            for idx, (fn, args, kwargs) in enumerate(jobs):
                futures[executor.submit(propagate_context_to_thread(fn), *args, **kwargs)] = idx
            pending = set(futures)
            deadline = timeout
            while pending:
                slice_timeout = min(_COUNCIL_POLL_INTERVAL_S, max(deadline, 0.0)) if self._agent is not None else deadline
                done, pending = _futures_wait(pending, timeout=slice_timeout)
                for future in done:
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        results[idx] = exc
                if self._agent is not None and getattr(self._agent, "_interrupt_requested", False) and pending:
                    interrupted = True
                    break
                deadline -= slice_timeout
                if deadline <= 0:
                    break
            for future in pending:
                future.cancel()
                results[futures[future]] = _INTERRUPTED_NOTE if interrupted else _STAGE_DEADLINE_NOTE
        finally:
            executor.shutdown(wait=not interrupted and not pending, cancel_futures=bool(pending) or interrupted)
        return results

    def _run_council(self, preset: dict[str, Any], member_view: list[dict[str, Any]],
                     members: list[dict[str, Any]]) -> dict[str, Any]:
        """Stages 1 + 2; returns the cacheable turn state."""
        stage_timeout = _stage_timeout(preset)
        member_temperature = _preset_temperature(preset, "member_temperature")
        cache_disabled, cache_ttl = _agent_cache_opts(self._agent)
        ctx_len_cache: dict[tuple[str, str], int | None] = {}
        worker_opts = dict(timeout=stage_timeout, context_length_cache=ctx_len_cache,
                           cache_disabled=cache_disabled, cache_ttl=cache_ttl)

        # ---- Stage 1: members answer in parallel ----
        jobs = [(functools.partial(_run_member), (slot, member_view),
                 {"temperature": member_temperature, **worker_opts}) for slot in members]
        raw = self._run_stage(jobs, timeout=stage_timeout)
        answers: list[dict[str, Any]] = []   # {label, slot, model_label, text}
        failures: list[str] = []
        accountings: list[Any] = []
        for slot, result in zip(members, raw):
            label = _council_slot_label(slot)
            if isinstance(result, BaseException):
                logger.warning("Council member %s failed: %s", label, result)
                failures.append(f"{label} [failed: {result}]")
                continue
            if isinstance(result, str):  # deadline / interrupt note
                failures.append(f"{label} {result}")
                continue
            text, acct = result
            accountings.append(acct)
            answers.append({
                "label": f"Response {_label_letter(len(answers))}",
                "slot": slot, "model_label": label, "text": text,
                "name": _slot_name(slot),
            })
        member_count = len(members)
        for idx, answer in enumerate(answers, start=1):
            self._emit("council.member", index=idx, count=len(answers),
                       label=answer["model_label"], name=answer["name"], text=answer["text"])

        # ---- Stage 2: anonymized peer review or dedicated judge ----
        reviews: list[dict[str, Any]] = []   # {reviewer_label, text, ranking}
        rankings: list[list[str]] = []
        all_labels = [a["label"] for a in answers]
        judge = _judge_slot(preset)
        run_stage2 = (
            not preset.get("lite")
            and len(answers) >= 2
            and not any(f.endswith(_INTERRUPTED_NOTE) for f in failures)
        )
        if run_stage2:
            # Anonymize: reviewers see labels only, never the member's model id.
            anonymized = [(a["label"], a["text"]) for a in answers]
            review_msgs = _review_messages(member_view, anonymized)
            if judge is not None:
                self._emit("council.phase", phase="review", members_done=len(answers),
                           members_total=member_count, chairman=_council_slot_label(judge))
                raw_reviews = self._run_stage(
                    [(functools.partial(_run_reviewer), (judge, "council_judge", review_msgs), dict(worker_opts))],
                    timeout=stage_timeout,
                )
                review_jobs = [(judge, "council_judge", raw_reviews[0])]
            else:
                self._emit("council.phase", phase="review", members_done=len(answers),
                           members_total=member_count,
                           chairman=_council_slot_label(preset.get("chairman") or {}))
                raw_reviews = self._run_stage(
                    [(functools.partial(_run_reviewer), (a["slot"], "council_review", review_msgs), dict(worker_opts))
                     for a in answers],
                    timeout=stage_timeout,
                )
                review_jobs = [(a["slot"], "council_review", result) for a, result in zip(answers, raw_reviews)]
            for idx, (slot, task, result) in enumerate(review_jobs, start=1):
                reviewer_label = _council_slot_label(slot)
                if isinstance(result, BaseException):
                    logger.warning("Council %s %s failed: %s", task, reviewer_label, result)
                    failures.append(f"{reviewer_label} ({task}) [failed: {result}]")
                    continue
                if isinstance(result, str):
                    failures.append(f"{reviewer_label} ({task}) {result}")
                    continue
                text, acct = result
                accountings.append(acct)
                ranking = parse_ranking(text)
                if ranking:
                    rankings.append(ranking)
                reviews.append({"reviewer_label": reviewer_label, "text": text, "ranking": ranking})
                self._emit(
                    "council.review", index=idx, count=len(review_jobs), reviewer=reviewer_label,
                    ranking=" > ".join(label.replace("Response ", "") for label in ranking),
                )
        self._fold_accountings(accountings)

        digest = aggregate_rankings(rankings, all_labels)
        interrupted = any(f.endswith(_INTERRUPTED_NOTE) for f in failures)
        return {
            "answers": answers, "failures": failures, "reviews": reviews,
            "digest": digest, "judge_used": judge is not None and run_stage2,
            "interrupted": interrupted,
        }

    # -- fan-out cache (turn-scoped, MoA "user_turn" semantics) ---------------

    def _fanout_cache_key(self, member_view: list[dict[str, Any]], members: list[dict[str, Any]]) -> tuple:
        """Hash the member-view prefix up to the LAST REAL user message (the synthetic
        advisory marker is skipped) so later tool iterations of the same turn are HITs."""
        last_user = next(
            (i for i in range(len(member_view) - 1, -1, -1)
             if member_view[i].get("role") == "user" and member_view[i].get("content") != _ADVISORY_INSTRUCTION),
            None,
        )
        sig_messages = member_view[: last_user + 1] if last_user is not None else member_view
        return (self.preset_name, _hash_messages(sig_messages), tuple(_council_slot_label(s) for s in members))

    # -- guidance ---------------------------------------------------------------

    def _build_guidance(self, state: dict[str, Any], chairman: dict[str, Any], stale: bool = False) -> str | None:
        answers = state.get("answers") or []
        failures = state.get("failures") or []
        if not answers and not failures:
            return None
        header = (
            "[LLM Council reference context]\n"
            f"Preset: {self.preset_name}\n"
            f"Chairman/acting model: {_council_slot_label(chairman or {})}\n"
        )
        parts: list[str] = [header]
        if not answers:
            parts.append(
                "All council members failed this turn — no advisory guidance is "
                "available. Act on your own judgment.\n"
            )
        digest = state.get("digest") or []
        if digest:
            label_to_model = {a["label"]: a["model_label"] for a in answers}
            lines = [
                f"{label_to_model.get(label, label)} — avg {avg:.2f} ({votes} votes)"
                for label, avg, votes in digest
            ]
            parts.append("FINAL RANKING digest:\n" + "\n".join(lines) + "\n")
        if failures:
            parts.append("Member/review failures:\n" + "\n".join(f"- {note}" for note in failures) + "\n")
        if answers:
            rendered = "\n\n".join(f"{a['label']} ({a['model_label']}):\n{a['text']}" for a in answers)
            parts.append("Member responses:\n" + rendered + "\n")
        reviews = state.get("reviews") or []
        if reviews:
            excerpts = []
            for review in reviews:
                ranking = "\n".join(
                    f"{pos}. {label}" for pos, label in enumerate(review.get("ranking") or [], start=1)
                ) or "(no parseable ranking)"
                excerpts.append(
                    f"Reviewer {review['reviewer_label']}:\nRanking:\n{ranking}\n{_clip(review.get('text') or '')}"
                )
            parts.append("Peer reviews:\n" + "\n\n".join(excerpts) + "\n")
        if stale:
            parts.append(_STALE_GUIDANCE_NOTE)
        parts.append(_CHAIRMAN_INSTRUCTION)
        return "\n".join(parts)

    # -- chairman (Stage 3) ------------------------------------------------------

    def _call_prepared_chairman(self, prepared: dict[str, Any], api_kwargs: dict[str, Any]) -> Any:
        """Send an already prepared Council chairman request exactly once (mirrors
        MoAChatCompletions._call_prepared_aggregator, including the alternation-rejection
        merge-and-retry-once path and the completed-response-as-stream-chunk fallback)."""
        from agent.moa_loop import MoAChatCompletions

        chairman = prepared["chairman"]
        if str(chairman.get("provider") or "").strip().lower() == "council":
            raise RuntimeError("Council chairman cannot be another Council preset")
        chairman_runtime = _slot_runtime(chairman)
        chairman_messages, tools = MoAChatCompletions._plan_aggregator_cache(
            self, prepared["messages"], api_kwargs.get("tools"), prepared.get("guidance"), chairman_runtime,
        )
        self._emit("council.synthesizing", chairman=_council_slot_label(chairman),
                   member_count=len(prepared.get("member_labels") or []) or None)
        self._emit("council.phase", phase="chairman",
                   members_done=prepared.get("members_done") or 0,
                   members_total=prepared.get("members_total") or 0,
                   chairman=_council_slot_label(chairman))
        # stream=True returns the RAW token stream (consumer reassembles + retries);
        # the non-streaming path forwards no stream/stream_options/timeout.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs = {"stream": True, "stream_options": api_kwargs.get("stream_options") or {"include_usage": True}}
            if api_kwargs.get("timeout") is not None:
                stream_kwargs["timeout"] = api_kwargs["timeout"]
        chairman_extra_body = _merge_slot_extra_body(
            chairman_runtime.pop("extra_body", None), api_kwargs.get("extra_body"))
        destination = destination_key(chairman_runtime)
        remembered = getattr(self, "_merge_same_role_destinations", None)
        if remembered is None:
            remembered = self._merge_same_role_destinations = set()
        merged = destination in remembered
        if merged:
            chairman_messages = merge_same_role_messages(chairman_messages)
        send = functools.partial(
            _call_llm, task="council_chairman", temperature=prepared["chairman_temperature"],
            max_tokens=api_kwargs.get("max_tokens"), tools=tools, extra_body=chairman_extra_body,
            reasoning_config=_slot_reasoning_config(chairman),  # slot pass-through only (no UI)
            **stream_kwargs, **chairman_runtime,
        )
        try:
            chairman_response = send(messages=chairman_messages)
        except Exception as exc:
            # Strict-alternation template rejected ``user(task), user(guidance)``: merge the pair
            # for THIS destination only and retry once; remember it so later iterations pre-merge.
            retry_messages = None if merged else merge_same_role_messages(chairman_messages)
            if retry_messages is None or retry_messages is chairman_messages or not is_role_alternation_rejection(exc, chairman_runtime):
                raise
            remembered.add(destination)
            logger.warning(
                "Council chairman %s rejected adjacent same-role messages — merging them for this "
                "destination for the rest of the session and retrying once: %.200s", _slot_label(chairman), exc,
            )
            chairman_messages = retry_messages
            chairman_response = send(messages=chairman_messages)
        if stream and hasattr(chairman_response, "choices"):
            # Some adapters return a completed response even when streaming was requested;
            # hand the loop a one-chunk iterator (same facade-boundary contract as MoA).
            return iter((_completed_response_as_stream_chunk(chairman_response),))
        return chairman_response

    # -- create ------------------------------------------------------------------

    def create(self, **api_kwargs: Any) -> Any:
        prepared_request = api_kwargs.pop("_moa_prepared_request", None)
        if prepared_request is not None:
            if not isinstance(prepared_request, dict):
                raise TypeError("_moa_prepared_request must be a dict")
            return self._call_prepared_chairman(prepared_request, api_kwargs)

        preset = _resolve_preset_cached(self.preset_name)
        messages = list(api_kwargs.get("messages") or [])
        chairman = preset.get("chairman") or {}
        # The Council path's virtual model/provider have no pricing entry; expose the real slot.
        self.last_chairman_slot = dict(chairman) if chairman else None
        chairman_temperature = _preset_temperature(preset, "chairman_temperature")
        if chairman_temperature is None and api_kwargs.get("temperature") is not None:
            chairman_temperature = api_kwargs.get("temperature")

        # A disabled preset = "use the chairman directly" (like a plain model).
        members = _enabled_members(preset) if preset.get("enabled", True) else []

        member_view = _reference_messages(messages)
        cache_key = self._fanout_cache_key(member_view, members)
        cache_hit = bool(cache_key == self._council_cache_key and self._council_cache_state)
        if cache_hit and self._council_cache_state is not None:
            # HIT: stages 1-2 already ran and were accounted this user turn.
            state = self._council_cache_state
        else:
            if members:
                state = self._run_council(preset, member_view, members)
            else:
                state = {"answers": [], "failures": [], "reviews": [], "digest": [],
                         "judge_used": False, "interrupted": False}
            # An interrupted fan-out is a partial snapshot: never cache it.
            if not state.get("interrupted"):
                self._council_cache_key = cache_key
                self._council_cache_state = state

        chairman_messages = [dict(m) for m in messages]
        guidance = self._build_guidance(
            state, chairman, stale=bool(cache_hit) and _tool_activity_since_last_user(messages),
        )
        if guidance:
            _attach_reference_guidance(chairman_messages, guidance)

        prepared_request = {
            "messages": chairman_messages, "guidance": guidance, "chairman": chairman,
            "chairman_temperature": chairman_temperature,
            "member_labels": [a["model_label"] for a in state.get("answers") or []],
            "members_done": len(state.get("answers") or []),
            "members_total": len(members),
        }
        if api_kwargs.pop("_moa_prepare_only", False):
            return prepared_request
        return self._call_prepared_chairman(prepared_request, api_kwargs)


class CouncilClient:
    """OpenAI-client-shaped wrapper: ``client.chat.completions`` is a ``CouncilChatCompletions``;
    the accounting surface is delegated to that facade (mirrors MoAClient).

    The SKIP flags follow the copilot-acp provider-plugin precedent: the facade
    already speaks OpenAI shapes end to end, so the auxiliary-client transport
    and async wrappers must not re-wrap it.
    """

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, preset_name: str, reference_callback: Any = None, agent: Any = None):
        self.chat = type("_CouncilChat", (), {})()
        self.chat.completions = CouncilChatCompletions(preset_name, reference_callback=reference_callback, agent=agent)

    def consume_member_usage(self) -> Any:
        return self.chat.completions.consume_member_usage()

    @property
    def last_chairman_slot(self) -> Any:
        return getattr(self.chat.completions, "last_chairman_slot", None)


# Relay table: event -> (primary kwarg, secondary kwarg or None, {cb kwarg: emit kwarg}).
# Callback signature mirrors MoA's relay: ``cb(event, primary, secondary, None, **council_*)``.
_COUNCIL_RELAY_EVENTS: dict[str, tuple[str, str | None, dict[str, str]]] = {
    "council.member": ("label", "text", {"council_index": "index", "council_count": "count",
                                         "council_name": "name"}),
    "council.review": ("reviewer", "ranking", {"council_index": "index", "council_count": "count",
                                               "council_ranking": "ranking"}),
    "council.phase": ("chairman", None, {"council_phase": "phase", "council_members_done": "members_done",
                                          "council_members_total": "members_total"}),
    "council.synthesizing": ("chairman", None, {"council_member_count": "member_count"}),
}


def build_council_facade(agent, preset_name: Any = None) -> CouncilClient:
    """Single construction point for ``CouncilClient`` (mirrors build_moa_facade): wires the
    ``council.*`` display relay through ``agent.tool_progress_callback`` at emit time."""

    def _council_reference_relay(event: str, **kwargs: Any) -> None:
        cb = getattr(agent, "tool_progress_callback", None)
        spec = _COUNCIL_RELAY_EVENTS.get(event)
        if cb is None or spec is None:
            return
        primary, secondary, extra_map = spec
        with contextlib.suppress(Exception):
            cb(
                event, str(kwargs.get(primary) or ""), str(kwargs.get(secondary) or "") if secondary else None, None,
                **{out: kwargs.get(src) for out, src in extra_map.items()},
            )

    resolved_preset = preset_name
    if resolved_preset is None and getattr(agent, "provider", None) == "council":
        resolved_preset = getattr(agent, "model", None)
    resolved_preset = str(resolved_preset or "default")
    try:
        from hermes_cli.config import load_config
        council_raw = load_config().get("council") or {}
        if resolved_preset not in (_normalize_council_presets(council_raw)):
            resolved_preset = str(council_raw.get("default_preset") or "default")
    except Exception:
        resolved_preset = "default"
    return CouncilClient(resolved_preset, reference_callback=_council_reference_relay, agent=agent)


def bind_council_runtime(agent, preset_name: Any, api_key: Any = None) -> None:
    """Make ``agent`` act as the Council preset: pin the virtual runtime fields and install
    the facade (mirrors bind_moa_runtime exactly)."""
    agent.model = str(preset_name or "default")
    agent.provider = agent.requested_provider = "council"
    agent.api_mode = "chat_completions"
    agent.api_key = api_key or "council-virtual-provider"
    agent.base_url = "council://local"
    agent._client_kwargs = {}
    agent.client = build_council_facade(agent, agent.model)
