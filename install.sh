#!/usr/bin/env bash
# Install the Council fork into a hermes-agent checkout.
# Usage: ./install.sh /path/to/hermes-agent
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "Usage: $0 /path/to/hermes-agent" >&2
  exit 1
fi
if [ ! -d "$TARGET/agent" ] || [ ! -f "$TARGET/agent/moa_loop.py" ]; then
  echo "Error: $TARGET does not look like a hermes-agent checkout (agent/moa_loop.py missing)." >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

cd "$TARGET"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "Error: $TARGET has uncommitted changes — commit or stash them first." >&2
  exit 1
fi

BRANCH="feature/council"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Branch $BRANCH already exists; checking it out."
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH"
fi

# Apply the squashed patch (idempotent: skip if already applied).
if git apply --check "$HERE/patches/council.patch" 2>/dev/null; then
  git apply "$HERE/patches/council.patch"
  git add -A
  git commit -m "feat(council): LLM Council virtual provider (karpathy/llm-council on MoA mechanics)"
  echo "Patch applied and committed on $BRANCH."
else
  echo "Patch does not apply cleanly (already applied, or checkout differs from main)."
  echo "Try instead:  git am $HERE/patches/000*.patch"
  exit 1
fi

echo
echo "Next steps:"
echo "  1. Rebuild the desktop app:  cd $TARGET/apps/desktop && npm run pack"
echo "  2. Restart Hermes, then:     Settings → Model → Council  (or: hermes council configure)"
echo "  3. Pick a model per seat, then select 'Council: default' in the model menu."
