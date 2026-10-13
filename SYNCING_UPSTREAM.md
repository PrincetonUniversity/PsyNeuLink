# Syncing the fork with upstream PsyNeuLink

`jonhanke-dev` is our integration branch. It carries changes upstream doesn't
have (the `_debugger` breakpoint/stepping hook, element-names propagation, the
dev `Makefile`, docs scaffolding). Upstream `PrincetonUniversity/PsyNeuLink`
`devel` moves on its own, so `jonhanke-dev` drifts behind and our builds — and
PsyNeuView, which is developed against this fork — get tested against an
increasingly stale PsyNeuLink. Regular syncs keep each catch-up small and catch
upstream breakage early. (See jonhanke-nam/PsyNeuLink#25.)

## Approach: merge, not rebase

`jonhanke-dev` is a published, shared branch. It is synced by **merging**
upstream `devel` in, never by rebasing:

- It's what PsyNeuView builds against and what teammates pull — rebasing
  rewrites every commit's SHA and forces everyone to hard-reset.
- The Mac app pins a PsyNeuLink commit SHA; a merge keeps old commits reachable
  (the pin stays valid until a deliberate bump), a rebase orphans them.
- Our changes are themselves PR merges, so a rebase would flatten that history
  or need fragile `--rebase-merges` that re-resolves conflicts on every replay.

Rebase still has a place — but only for a short-lived *private* feature branch
(branched off `jonhanke-dev`, not yet pushed), to tidy it before its PR lands.

## How to sync

```
make sync-upstream                 # merge, smoke-test, open a PR for review
make sync-upstream ARGS="--dry-run"  # just report drift + planned actions
```

`bin/sync-upstream-devel.sh` does the work:

1. Preflight (clean tree, remotes present) and fetch `origin` + `jonhanke-nam`.
2. Branch `sync/upstream-devel-<timestamp>` from `jonhanke-nam/jonhanke-dev`.
3. `git merge origin/devel` (merge commit). On conflict it stops and lists the
   conflicted paths — resolve on the sync branch, then re-run the smoke tests by
   hand and push. Watch the `_debugger` hook surface and anything touching
   composition execution.
4. **Smoke gate** — must pass before anything is pushed:
   - `tests/misc/test_debugger.py` — the hook still works after the merge.
   - `tests/misc/test_element_names*.py` — element-names additions intact.
   - `tests/composition/test_emcomposition.py::TestExecution::test_simple_execution_without_learning`
     — the upstream-integration canary (clean pull + core execution).
5. Push the sync branch and open a PR into `jonhanke-dev`. The PR is **reviewed,
   not auto-merged** — same human-review gate as the rest of the workflow.
6. After the PR merges, **bump the Mac app's pinned PsyNeuLink commit** to the
   new `jonhanke-dev` HEAD as its own reviewed change. Never let the pin drift
   silently.

## Cadence

Weekly, and always before a PsyNeuView release cut.

## Later

A scheduled GitHub Action can call the same script to open the sync PR with no
local step. Deferred until the manual procedure has run a few times.
