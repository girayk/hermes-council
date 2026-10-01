# Hermes Council

An **LLM Council** for [Hermes Agent](https://github.com/NousResearch/hermes-agent) —
[karpathy/llm-council](https://github.com/karpathy/llm-council) semantics built on
Hermes's own Mixture-of-Agents mechanics.

Four fixed advisors answer your question in parallel, **anonymously review and rank
each other**, and a chairman model synthesizes the final answer — while staying a
normal Hermes model: tools, streaming, sessions, interrupts, everything works.

```
Your question
   │
   ├─ Stage 1 · First opinions (parallel, each seat steered by its duty)
   │    First-Principles Analyst · Skeptic · Pragmatist · Researcher
   │
   ├─ Stage 2 · Peer review (anonymized as "Response A/B/C/D")
   │    every seat ranks the others — or a dedicated Judge ranks them —
   │    strict "FINAL RANKING:" parsing → aggregate avg rank + votes
   │
   └─ Stage 3 · Chairman acts
        ranked digest + all answers as private guidance;
        the chairman is the acting model (tools + streaming)
```

## Why it's a core fork, not a plugin

Council is implemented exactly like Hermes's built-in **Mixture of Agents** feature —
a *virtual provider* whose presets appear in the model menu, an OpenAI-shaped facade
client the agent loop calls, and member calls that borrow your already-configured
provider credentials via `call_llm` (**no extra API keys, ever**). Those surfaces
(agent loop dispatch, model menu, Settings page) live in the Hermes core, so Council
ships as a patch set against `hermes-agent`, not a marketplace plugin. MoA itself is
untouched — the fork is purely additive.

## The council

| Seat | Duty (injected into its Stage-1 prompt) |
|---|---|
| **First-Principles Analyst** | Reason from fundamentals; ignore convention and popularity. |
| **Skeptic** | Assume the obvious answer is wrong; hunt for failure modes, hidden costs and base rates. |
| **Pragmatist** | What actually works in practice — constraints, effort, reversibility, second-order effects. |
| **Researcher** | Ground claims in verifiable evidence; use web search and code inspection where available. |

Seats are fixed — you only pick **which of your models sits in each seat**.
Stage 2 is fully anonymous (reviewers never see seat names, roles, or model ids —
karpathy's anti-favoritism rule, enforced by tests). Unassigned seats are skipped.

Optional slots:
- **Judge** — when set, one model ranks the anonymized answers instead of peer review.
- **Chairman** — the acting model. It runs every step of the tool loop and carries
  almost all of the cost (same billing model as MoA's aggregator).

## Install

Requirements: a checkout of [hermes-agent](https://github.com/NousResearch/hermes-agent)
and the Hermes desktop app already working (this patch adds to both).

```bash
git clone https://github.com/<you>/hermes-council.git
cd hermes-council
./install.sh /path/to/hermes-agent      # applies patches/council.patch
```

Or by hand:

```bash
cd /path/to/hermes-agent
git checkout -b feature/council
git am /path/to/hermes-council/patches/*.patch
```

Then rebuild the desktop app (`cd apps/desktop && npm run pack`) and restart Hermes.

## Configure

**Desktop:** Settings → Model → **Council** — pick a model for each of the 4 seats,
optionally a Judge, and the Chairman. Autosaves.

**CLI:**

```bash
hermes council configure     # walks the 4 seats + judge + chairman
hermes council list          # shows the roster
```

**config.yaml** (what the UI writes):

```yaml
council:
  default_preset: default
  presets:
    default:
      members:                       # the 4 fixed seats, positionally
        - {provider: qwen, model: qwen3.8-max, enabled: true}      # First-Principles Analyst
        - {provider: qwen, model: deepseek-v4-pro, enabled: true}   # Skeptic
        - {provider: qwen, model: glm-5.3, enabled: true}           # Pragmatist
        - {provider: "", model: "", enabled: false}                 # Researcher (unassigned)
      judge: {provider: "", model: "", enabled: true}   # empty = peer review
      chairman: {provider: qwen, model: qwen3.8-max}    # acting model
      lite: false                                       # true = skip Stage 2 (N+1 calls)
      enabled: true
```

## Use it

Select **Council: default** in the model menu (the row shows the seat names), or:

```bash
/model default --provider council
hermes -z "your hard question" --model default --provider council
```

While a council turn runs you get live progress in the chat's reasoning area and the
TUI: each member's answer as it lands (`council.member`), each reviewer's ranking
(`council.review`), and the chairman handoff (`council.synthesizing`). The chairman's
synthesis streams out as the normal assistant reply — tool calls included.

## Cost

Per user turn: `N+1` calls with `lite: true` (N seat answers + chairman),
up to `2N+1` with peer review (N answers + N reviews + chairman). The fan-out runs
**once per user turn** — tool-loop iterations reuse the cached guidance and only
re-call the chairman, exactly like MoA's default `user_turn` cadence. Every stage
has hard wall-clock deadlines; a wedged model is dropped with a failure note,
never hangs the turn.

## What's in the patch set

| Area | Files |
|---|---|
| Engine (3-stage loop, facade, fan-out cache, deadlines) | `agent/council_loop.py` + council dispatch at every `provider == "moa"` gate in `agent/` |
| Config/CLI | `hermes_cli/council_config.py`, `council_cmd.py`, `subcommands/council.py` |
| Menu/catalog | virtual `council` row in `inventory.py`, `models_catalog_static.py`, `model_switch*.py`, `auth.py`, `runtime_provider.py` |
| HTTP API | `GET/PUT /api/model/council` in `web_routers/models.py` |
| Progress events | `council.member/review/phase/synthesizing` in `tui_gateway/` + gateway contract |
| Desktop UI | Settings → Model → Council page, "Council presets" menu section with seat names, live stage rendering, i18n |
| Tests | `tests/agent/test_council_loop.py` (16), `tests/hermes_cli/test_council_config.py` (18) + MoA regression suites green |

MoA is untouched: all Council code paths are additive (`provider in ('moa','council')`
or parallel branches), and the full MoA test suite passes unchanged.

## Credits

- Council protocol: [karpathy/llm-council](https://github.com/karpathy/llm-council)
- Seat roster & anti-sycophancy duties: the `llm-council` skill protocol
- Mechanics: Hermes Agent's Mixture-of-Agents (`agent/moa_loop.py`)
