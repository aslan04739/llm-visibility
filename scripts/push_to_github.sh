#!/usr/bin/env bash
set -euo pipefail

# Simple helper to commit & push current changes to 'origin' if configured.
MSG="${1:-Update llm-visibility}"

git add .
if git diff --staged --quiet; then
  echo "No changes to commit."
else
  git commit -m "$MSG" || echo "Commit failed or nothing to commit."
fi

if git remote | grep -q '^origin$'; then
  git push origin HEAD
else
  echo "No remote named 'origin' configured. Run 'git remote add origin <url>' then push manually."
fi

echo "Done."
