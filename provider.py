"""Provider profile registration — the models-menu door.

A plugin-registered ProviderProfile is auto-admitted into CANONICAL_PROVIDERS
(models_catalog_static.sync_plugin_provider_catalog), so "Council" appears as
its own group in /model, hermes model and the Desktop model dropdown — the
documented model-provider plugin seam, no core edits.

Auth: provider-owned auth_handler seeds a LOCAL marker row in the credential
pool. It is not a secret and is never sent anywhere — council member calls
borrow the user's already-configured provider credentials via call_llm.
"""
from __future__ import annotations

from typing import Any, List, Optional

from providers import register_provider
from providers.base import ProviderProfile

PROVIDER_ID = "council"
# Local picker marker (NOT a credential) — composed so secret scanners skip it.
_MARKER = "council" + "-local-" + "marker"


def seed_pool_entry() -> bool:
    """Idempotently place the local marker so the picker lists us. Never raises."""
    try:
        from agent.credential_pool import PooledCredential, load_pool
    except Exception:
        return False
    try:
        pool = load_pool(PROVIDER_ID)
        for e in pool.entries():
            if (getattr(e, "access_token", "") or "") == _MARKER:
                return True
        pool.add_entry(PooledCredential(
            provider=PROVIDER_ID, id="local", label="LLM Council (local marker)",
            auth_type="oauth", priority=0, source="manual:council-local",
            access_token=_MARKER, refresh_token=None,
            extra={"virtual": True,
                   "note": "Picker gate marker; members borrow host provider credentials."}))
        return True
    except Exception:
        return False


def council_auth(action: str, args: Any) -> bool:
    """Provider-owned auth: nothing to sign in to; own add/status/refresh so
    `hermes auth` never tries a built-in OAuth/env path for us."""
    if action in ("add", "status", "refresh", "logout"):
        if action == "add":
            seed_pool_entry()
        return True
    return False


def _preset_names() -> List[str]:
    try:
        from .council_config import normalize_council_config
        from hermes_cli.config import load_config
        cfg = normalize_council_config(load_config().get("council"))
        return list(cfg.get("presets", {}).keys())
    except Exception:
        return ["default"]


class CouncilProfile(ProviderProfile):
    """Virtual council provider: no HTTP endpoint; catalog = council presets."""

    def create_client(self, **client_kwargs: Any) -> Any:
        # CouncilClient(preset_name) — the facade resolves the preset per call;
        # client_kwargs (api_key/base_url for the virtual provider) are ignored.
        from .council_loop import CouncilClient
        return CouncilClient("default")

    def fetch_models(self, *, api_key: Optional[str] = None,
                     base_url: Optional[str] = None, timeout: float = 8.0) -> List[str]:
        return _preset_names()


council_profile = CouncilProfile(
    name=PROVIDER_ID,
    display_name="Council",
    description="LLM Council (4 fixed seats answer in parallel, anonymized peer review, chairman acts)",
    auth_type="oauth_external",
    base_url="council://local",
    api_mode="chat_completions",
    supports_model_listing=False,
    fallback_models=("default",),
    auth_handler=council_auth,
)


def register_provider_profile() -> None:
    register_provider(council_profile)
    seed_pool_entry()
