# hermes-council

An **LLM Council** plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) —
[karpathy/llm-council](https://github.com/karpathy/llm-council) semantics as a first-class
Hermes plugin.

Four fixed advisor seats answer your question **in parallel**, **anonymously review and
rank each other**, and a **Chairman** model synthesizes the final answer. Members borrow
the provider credentials you already configured in Hermes — **no API keys, no extra setup**.

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
   └─ Stage 3 · Chairman synthesis
        ranked digest + all answers as private guidance
```

## The council

| Seat | Duty (injected into its Stage-1 prompt) |
|---|---|
| **First-Principles Analyst** | Reason from fundamentals; ignore convention and popularity. |
| **Skeptic** | Assume the obvious answer is wrong; hunt for failure modes, hidden costs and base rates. |
| **Pragmatist** | What actually works in practice — constraints, effort, reversibility, second-order effects. |
| **Researcher** | Ground claims in verifiable evidence; use web search and code inspection where available. |

Seats are fixed — you only pick **which of your models sits in each seat** (panel
or `hermes council configure`). Stage 2 is fully anonymous: reviewers never see seat
names, roles, or model ids (karpathy's anti-favoritism rule, enforced by tests).
Unassigned seats are skipped.

Optional slots:

- **Judge** — when set, one model ranks the anonymized answers in a single call instead
  of full peer review.
- **Chairman** — synthesizes the final answer (required).

## Install

```bash
hermes plugins install girayk/hermes-council
hermes plugins enable hermes-council
```

or clone into `~/.hermes/plugins/`:

```bash
git clone https://github.com/girayk/hermes-council ~/.hermes/plugins/hermes-council
hermes plugins enable hermes-council
```

Requires a working Hermes install with at least two configured provider models
(any providers — Qwen, OpenRouter, Anthropic, OpenAI, local…). Restart the
desktop app / gateway after enabling.

## Configure

**Desktop:** the plugin ships a **/council** panel (sidebar → Council): each seat shows
its name + duty, you pick provider/model from your configured catalog; Judge and
Chairman below; autosaves to `council:` in config.yaml.

**CLI:**

```bash
hermes council configure    # walk the 4 seats + judge + chairman
hermes council list         # show the roster
```

**config.yaml** (what the panel writes):

```yaml
council:
  default_preset: default
  presets:
    default:
      members:                     # the 4 fixed seats, positionally
        - {provider: qwen, model: qwen3.8-max, enabled: true}       # First-Principles Analyst
        - {provider: qwen, model: deepseek-v4-pro, enabled: true}   # Skeptic
        - {provider: qwen, model: glm-5.3, enabled: true}           # Pragmatist
        - {provider: "", model: "", enabled: false}                 # Researcher (unassigned)
      judge: {provider: "", model: "", enabled: true}   # empty = peer review
      chairman: {provider: qwen, model: qwen3.8-max}
      lite: false                                        # true = skip Stage 2
      enabled: true
```

## Use it

Three surfaces, one engine:

1. **Tool** — the model can call `council_deliberate` on hard questions (the bundled
   skill teaches it when that's worth the cost).
2. **Slash command** — `/council <question>` in any chat (CLI, TUI, gateway).
3. **Model menu** — the plugin registers a **Council** group in every model picker
   (its "model" is the council preset): `/model default --provider council`, or the
   desktop model dropdown. Every turn of that session runs through the council — the
   chairman acts as the model (tools + streaming work normally), with the ranked
   member digest as its private guidance.

Recent deliberations land in `~/.hermes/council/runs/` and show in the panel.

## Cost

Per deliberation: `N+1` calls with `lite: true` (N seat answers + chairman), up to
`2N+1` with peer review. As a session model, the fan-out runs **once per user turn**;
tool-loop iterations reuse the cached guidance and only re-call the chairman. Every
stage has hard wall-clock deadlines — a wedged model is dropped with a failure note,
never hangs your turn.

## How it works (plugin surfaces only — no core patches)

| Surface | Where |
|---|---|
| Model menu group | plugin `ProviderProfile` via `register_provider` — the documented model-provider plugin seam; virtual `council://local` provider, presets as models |
| Turn engine | `CouncilClient` OpenAI-shaped facade via `create_client()`; stages call `agent.auxiliary_client.call_llm(task=…, provider=…, model=…)` with YOUR credentials |
| Tool / slash / CLI | `register(ctx)`: `council_deliberate`, `/council`, `hermes council` |
| Desktop panel | `desktop/plugin.js` (plain ESM, plugin SDK) + `dashboard/plugin_api.py` routes under `/api/plugins/hermes-council/` |
| Skill | bundled `skills/llm-council/SKILL.md` (when to convene a council, anti-sycophancy rules) |
| Config | `council:` block in config.yaml (`council_config.py` normalize/validate/save) |

Stage-2 prompts contain only "Response A/B/C" labels — seat names, roles and model
ids never reach reviewers (anonymity-guard tests). The chairman's guidance
de-anonymizes: `Response A (First-Principles Analyst (qwen:qwen3.8-max))`.

## Development

```bash
hermes plugins validate ~/.hermes/plugins/hermes-council   # manifest, security scan, desktop surface
cd tests && HERMES_HOME=/tmp/council-test pytest -q        # 35 tests (engine, config, anonymity guards)
```

## Credits

- Council protocol: [karpathy/llm-council](https://github.com/karpathy/llm-council)
- Seat roster & duties: the `llm-council` skill protocol (independence before interaction,
  stake-weighted criticism, anti-sycophancy rules)
- Mechanics follow Hermes's own Mixture-of-Agents design (virtual provider + facade
  client + `call_llm` fan-out), implemented entirely through public plugin seams.
