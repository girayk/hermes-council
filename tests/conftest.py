"""Load the hermes-council plugin package (hyphenated dir name) for tests.

hermes-agent must be importable: set HERMES_AGENT_SRC=/path/to/hermes-agent,
or rely on the default ~/.hermes/hermes-agent location.
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("HERMES_HOME", str(Path.home() / ".hermes"))

# Locate the hermes-agent source tree (env override, then the standard install).
for _cand in (os.environ.get("HERMES_AGENT_SRC"),
              str(Path.home() / ".hermes" / "hermes-agent")):
    if _cand and (Path(_cand) / "agent" / "moa_loop.py").is_file():
        sys.path.insert(0, _cand)
        break
else:
    raise RuntimeError(
        "hermes-agent source not found; set HERMES_AGENT_SRC=/path/to/hermes-agent")

PKG_DIR = Path(__file__).resolve().parent.parent

# Load at import time (before test collection resolves `hermes_council.*`).
if "hermes_council" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "hermes_council", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_council"] = _mod
    _spec.loader.exec_module(_mod)
