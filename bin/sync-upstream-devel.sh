#!/usr/bin/env bash
#
# Sync the fork's integration branch with upstream PsyNeuLink devel.
#
# Merges upstream devel onto a fresh `sync/upstream-devel-<timestamp>` branch,
# runs a smoke gate, and opens a PR into the integration branch. Never pushes
# directly to the integration branch and never merges the PR — a human reviews
# it (see SYNCING_UPSTREAM.md and jonhanke-nam/PsyNeuLink#25).
#
# Usage:
#   bin/sync-upstream-devel.sh [--dry-run] [--no-pr] [--skip-smoke]
#
#   --dry-run     Report drift and the planned actions, then stop before any
#                 branch/merge/push. Read-only.
#   --no-pr       Do the merge + smoke + push the sync branch, but don't open
#                 the PR (print the branch name and PR URL instead).
#   --skip-smoke  Skip the test gate. Escape hatch only — defeats the point.
#
# Overridable via environment:
#   UPSTREAM_REMOTE (origin)  UPSTREAM_BRANCH (devel)
#   FORK_REMOTE (jonhanke-nam)  INTEGRATION_BRANCH (jonhanke-dev)
#   VENV_DIR (.venv)
#
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-origin}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-devel}"
FORK_REMOTE="${FORK_REMOTE:-jonhanke-nam}"
INTEGRATION_BRANCH="${INTEGRATION_BRANCH:-jonhanke-dev}"
VENV_DIR="${VENV_DIR:-.venv}"
GH_REPO="jonhanke-nam/PsyNeuLink"

DRY_RUN=0
OPEN_PR=1
RUN_SMOKE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1 ;;
    --no-pr)      OPEN_PR=0 ;;
    --skip-smoke) RUN_SMOKE=0 ;;
    -h|--help)    tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- Preflight -------------------------------------------------------------
command -v git >/dev/null || fail "git not found"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not in a git repo"
cd "$REPO_ROOT"

git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 || fail "missing remote '$UPSTREAM_REMOTE'"
git remote get-url "$FORK_REMOTE"     >/dev/null 2>&1 || fail "missing remote '$FORK_REMOTE'"

[[ -z "$(git status --porcelain)" ]] || fail "working tree is dirty — commit/stash first (the sync checks out a new branch here)"

say "Fetching $UPSTREAM_REMOTE and $FORK_REMOTE"
git fetch --quiet "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
git fetch --quiet "$FORK_REMOTE" "$INTEGRATION_BRANCH"

BASE_REF="$FORK_REMOTE/$INTEGRATION_BRANCH"
UPSTREAM_REF="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

# --- Drift report ----------------------------------------------------------
read -r AHEAD BEHIND < <(git rev-list --left-right --count "$BASE_REF...$UPSTREAM_REF")
say "Drift"
printf '  %s is %s ahead / %s behind %s\n' "$INTEGRATION_BRANCH" "$AHEAD" "$BEHIND" "$UPSTREAM_REF"

if [[ "$BEHIND" -eq 0 ]]; then
  echo "Already up to date with upstream — nothing to sync."
  exit 0
fi

SYNC_BRANCH="sync/upstream-devel-$(date +%Y%m%d-%H%M%S)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "Dry run — planned actions"
  cat <<PLAN
  1. branch  $SYNC_BRANCH  from  $BASE_REF
  2. merge   $UPSTREAM_REF  (merge commit)
  3. smoke   debugger + element-names + EM baseline tests
  4. push    $SYNC_BRANCH -> $FORK_REMOTE
  5. PR      $SYNC_BRANCH -> $INTEGRATION_BRANCH  (review, do not auto-merge)
  6. reminder: bump the Mac app PsyNeuLink pin after the PR merges
PLAN
  exit 0
fi

# --- Merge -----------------------------------------------------------------
say "Creating $SYNC_BRANCH from $BASE_REF"
git checkout -b "$SYNC_BRANCH" "$BASE_REF"

say "Merging $UPSTREAM_REF"
if ! git merge --no-edit "$UPSTREAM_REF"; then
  echo >&2
  echo "Merge conflicts — resolve them on '$SYNC_BRANCH', then re-run the smoke" >&2
  echo "tests by hand and push. Conflicted paths:" >&2
  git diff --name-only --diff-filter=U | sed 's/^/  /' >&2
  exit 1
fi

# --- Smoke gate ------------------------------------------------------------
if [[ "$RUN_SMOKE" -eq 1 ]]; then
  PYTEST="$VENV_DIR/bin/pytest"
  [[ -x "$PYTEST" ]] || fail "no pytest at $PYTEST — run 'make install-dev' (or set VENV_DIR) first"

  say "Smoke gate"
  declare -a FAILED=()
  run_bucket() {
    local name="$1"; shift
    printf -- '--- %s\n' "$name"
    if "$PYTEST" -q "$@"; then
      printf -- '--- %s: PASS\n' "$name"
    else
      printf -- '--- %s: FAIL\n' "$name"
      FAILED+=("$name")
    fi
  }
  # Fork-surface tests: did the upstream merge break our additions?
  run_bucket "debugger hook"   tests/misc/test_debugger.py
  run_bucket "element names"   tests/misc/test_element_names.py \
                               tests/misc/test_element_names_mdf.py \
                               tests/misc/test_element_names_propagation.py
  # Upstream-integration canary: clean pull + core execution still works.
  run_bucket "EM baseline"     "tests/composition/test_emcomposition.py::TestExecution::test_simple_execution_without_learning"

  if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo >&2
    echo "Smoke gate FAILED: ${FAILED[*]}" >&2
    echo "Sync branch '$SYNC_BRANCH' left in place for inspection. Not pushing." >&2
    exit 1
  fi
  SMOKE_LINE="all green (debugger, element-names, EM baseline)"
else
  SMOKE_LINE="SKIPPED (--skip-smoke)"
fi

# --- Push + PR -------------------------------------------------------------
say "Pushing $SYNC_BRANCH -> $FORK_REMOTE"
git push -u "$FORK_REMOTE" "$SYNC_BRANCH"

PR_BODY="Automated sync of \`$UPSTREAM_REF\` into \`$INTEGRATION_BRANCH\` (Refs #25).

- Drift before sync: $AHEAD ahead / $BEHIND behind upstream.
- Smoke gate: $SMOKE_LINE.

After this merges, bump the Mac app's pinned PsyNeuLink commit to the new \`$INTEGRATION_BRANCH\` HEAD as its own reviewed change — don't let the pin drift silently.

Generated by \`bin/sync-upstream-devel.sh\`."

if [[ "$OPEN_PR" -eq 1 ]] && command -v gh >/dev/null 2>&1; then
  say "Opening PR -> $INTEGRATION_BRANCH"
  gh pr create -R "$GH_REPO" --base "$INTEGRATION_BRANCH" --head "$SYNC_BRANCH" \
    --title "Sync $INTEGRATION_BRANCH with upstream devel ($(date +%Y-%m-%d))" \
    --body "$PR_BODY"
else
  say "PR not opened"
  echo "Branch pushed: $SYNC_BRANCH"
  echo "Open a PR into $INTEGRATION_BRANCH at:"
  echo "  https://github.com/$GH_REPO/pull/new/$SYNC_BRANCH"
fi

say "Done"
echo "Reminder: after the PR merges, bump the Mac app PsyNeuLink pin to the new $INTEGRATION_BRANCH HEAD."
