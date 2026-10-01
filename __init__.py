"""hermes-council — LLM Council plugin (karpathy/llm-council on Hermes).

General plugin surface (docs: Build a Hermes Plugin):
  * register(ctx) — council_deliberate tool, /council slash command,
    `hermes council` CLI, bundled skill.
  * provider.py  — a plugin ProviderProfile ("Council" group in every model
    menu; presets are the selectable models; create_client returns the
    CouncilClient facade so a selected council preset drives a whole turn).

Members borrow the user's already-configured provider credentials through
agent.auxiliary_client.call_llm — this plugin handles no API keys.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Register the provider profile at import: provider discovery imports this
# package from $HERMES_HOME/plugins, and the "Council" menu row must exist in
# every process that resolves providers (CLI, gateway, desktop backend).
from .provider import register_provider_profile  # noqa: E402

try:
    register_provider_profile()
except Exception:
    logger.warning("hermes-council: provider registration failed", exc_info=True)


# ── tool: council_deliberate ──────────────────────────────────────────────────

COUNCIL_DELIBERATE_SCHEMA = {
    "name": "council_deliberate",
    "description": (
        "Convene the LLM Council for a hard question: four fixed advisor seats "
        "(First-Principles Analyst, Skeptic, Pragmatist, Researcher) answer in "
        "parallel using models the user configured, then anonymously review and "
        "rank each other's answers, and the chairman model synthesizes one final "
        "answer. Use for high-stakes, contested, or ambiguous questions where "
        "cross-checking matters. Expensive: up to 2N+1 model calls — reserve for "
        "genuinely difficult queries, never for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to put to the council."},
        },
        "required": ["question"],
    },
}


def handle_council_deliberate(args: dict, **kwargs) -> str:
    """Run one council deliberation. Returns a JSON string (never raises).

    Uses the full facade (create) — stages 1-2 fan-out plus the chairman
    synthesis — exactly like a council-driven agent turn, minus tools.
    """
    try:
        question = str(args.get("question") or "").strip()
        if not question:
            return json.dumps({"error": "No question provided."})
        from .council_loop import CouncilClient
        from . import runstore

        client = CouncilClient("default")
        resp = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": question}],
            stream=False,
        )
        from agent.moa_loop import _extract_text
        answer = _extract_text(resp)
        chairman = getattr(client, "last_chairman_slot", None) or {}
        run_id = runstore.save_deliberation(question, answer, chairman)
        chair_label = (f"{chairman.get('provider')}:{chairman.get('model')}"
                       if chairman.get("provider") else "chairman")
        return json.dumps({
            "final_answer": answer,
            "chairman": chair_label,
            "run_id": run_id,
        }, ensure_ascii=False)
    except Exception as exc:
        logger.warning("council_deliberate failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})


# ── slash command: /council ───────────────────────────────────────────────────

def _handle_council_slash(raw_args: str) -> str:
    question = (raw_args or "").strip()
    if not question:
        return ("Usage: /council <question>\n"
                "Runs the question through the LLM Council (4 fixed seats answer, "
                "anonymized peer review, chairman synthesizes).")
    result = handle_council_deliberate({"question": question})
    try:
        data = json.loads(result)
    except Exception:
        return result
    if data.get("error"):
        return f"Council run failed: {data['error']}"
    lines = [data.get("final_answer") or "(no answer)"]
    if data.get("chairman"):
        lines.append(f"\n[council · chairman: {data['chairman']}]")
    return "\n".join(lines)


# ── CLI: hermes council list|configure ────────────────────────────────────────

def _cli_setup(sub):
    subs = sub.add_subparsers(dest="council_command")
    subs.add_parser("list", aliases=["ls"], help="Show the council roster")
    subs.add_parser("configure", aliases=["config"], help="Pick a model for each seat")


def _cli_handler(args) -> int:
    from .council_cmd import cmd_council
    cmd_council(args)
    return 0


# ── registration ──────────────────────────────────────────────────────────────

def register(ctx):
    # The menu door, again, from the general-plugin path too (idempotent).
    try:
        register_provider_profile()
    except Exception:
        pass

    ctx.register_tool(
        name="council_deliberate",
        toolset="council",
        schema=COUNCIL_DELIBERATE_SCHEMA,
        handler=handle_council_deliberate,
    )
    ctx.register_command(
        "council",
        handler=_handle_council_slash,
        description="Run a question through the LLM Council (4 seats, peer review, chairman)",
        args_hint="<question>",
    )
    ctx.register_cli_command(
        name="council",
        help="LLM Council — list the roster, configure seat models",
        setup_fn=_cli_setup,
        handler_fn=_cli_handler,
    )
    from pathlib import Path
    skill_md = Path(__file__).parent / "skills" / "llm-council" / "SKILL.md"
    if skill_md.exists():
        try:
            ctx.register_skill("llm-council", skill_md)
        except Exception:
            pass
