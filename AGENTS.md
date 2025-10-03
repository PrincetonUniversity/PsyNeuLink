PsyNeuLink Agent Guide (AGENTS.md)

Scope: This file applies to the entire repository. It provides practical guidance for human and AI agents contributing code, tests, and docs to PsyNeuLink. Follow these instructions for any files you modify within this repo.

**Quick Start**
- Use Python 3.12+.
- Create a virtual environment, then install in editable mode with dev tools: `pip install -e .[dev]`.
- Test suite is massive and takes over one hour to run, only run targeted tests with pytest when checking code.
- Keep changes surgical; update docs and tests alongside code.

**Repository Map**
- `psyneulink/core`: Core components (Mechanisms, Projections, Ports, Functions), execution/runtime, registries.
- `psyneulink/library`: Extensions and example compositions/models built on core.
- `tests/`: Pytest suite organized by area (components, projections, ports, llvm, models, misc, api, json, mdf, etc.).
- `docs/`: Sphinx documentation sources; reStructuredText and project config.
- `tutorial/`: Jupyter tutorial notebook and extras.
- `Scripts/`, `Matlab/`, `ys_test/`: Prototypes, historical scripts, or external assets; tests ignore `Scripts/`.
- Top-level: `CONVENTIONS.md` (style rules), `CONTRIBUTING.md`, `setup.cfg` (pytest, coverage, lint), requirements files.
- Do not make commits without first checking!

**Environment & Installation**
- Python: `python>=3.12`. Though Python 3.9 is supported.
- Virtual .venv in root of repo.
- Editable install: `uv pip install -e .`.
- Extras:
  - Dev: `uv pip install -e .[dev]` (pytest, coverage, xdist, style plugins).
  - Docs: `uv pip install -e .[doc]`.
  - Tutorial: `uv pip install -e .[tutorial]`.
  - CUDA: `uv pip install -e .[cuda]` (requires supported GPU stack).
- Respect pinned dependency ranges in `requirements.txt` (e.g., `llvmlite`, `torch`, `numpy`, `matplotlib`). Avoid bumping pins as part of unrelated changes.

**Running Tests**
- Default suite (parallel by xdist is configured in `setup.cfg`): `pytest`.
- By area: `pytest tests/components -q`, `pytest tests/ports -q`, etc.
- Single test: `pytest path/to/test_file.py::test_name -n 0 -q`.
- Markers (selected via `-m`):
  - `llvm`: Requires LLVM runtime compiler support.
  - `cuda`: Requires CUDA; always combined with `llvm`.
  - `pytorch`: Requires PyTorch availability.
  - `stress`: Long-running; skipped by default unless `--stress` is passed.
- Lint and docstyle run inside pytest (pycodestyle, pydocstyle). Fix failures before committing.
- Coverage is configured in `setup.cfg` with fail-under threshold; run with `pytest --cov psyneulink -q` when needed.
- If debugging nondeterminism, tests can be run single-threaded with `-n 0`. Seeds are set in `conftest.py` for determinism.

**Coding Conventions**
- Always follow `CONVENTIONS.md`. Key highlights:
  - Naming: Classes `CamelCase`; instance attributes/methods `snake_case`; constants `UPPER_SNAKE`.
  - Component names end with their type (e.g., `TransferMechanism`, `LearningProjection`).
  - Public classes/components are referred to in caps within docs (e.g., Mechanism, Projection, Port).
  - Errors vs warnings: use `raise <Class>Error("PROGRAM ERROR: ...")` for disallowed/bug states; `warnings.warn("WARNING: ...")` for user-action caveats.
  - Avoid bare `assert`s in production paths; they are reserved for tests or temporary development (not allowed on `master`).
  - Module layout: license, module docstring, imports, constants/keywords, error classes, factory (if any), main class with standard method order.

**Docstrings & Docs**
- Use reStructuredText docstrings and the standard section structure from `CONVENTIONS.md` for modules/classes:
  - Overview; Creating a(n) X; Structure; Execution; Class Reference.
- Keep constructor arguments documented in “Arguments” (Parameters) in the same order as the constructor; end with `params`, `name`, `prefs` if applicable.
- Build docs locally:
  - Install: `uv pip install -e .[doc]`.
  - HTML: `make -C docs html` (or `sphinx-build -b html docs/source docs/build`).
  - Add new `.rst` pages under `docs/source/` and connect them via toctrees as appropriate.

**Adding or Modifying Code**
- Place new core Components under the appropriate subpackage:
  - Mechanisms: `psyneulink/core/components/mechanisms/...`
  - Projections: `psyneulink/core/components/projections/...`
  - Ports & Signals: `psyneulink/core/components/ports/...`
  - Functions: `psyneulink/core/components/functions/...`
  - Composition/runtime-related: `psyneulink/library/compositions/...` or core scheduling/runtime modules.
- Expose new public classes:
  - Update the relevant `__init__.py` to import and add to `__all__` so they are available from the top-level package if intended.
- Interop and JIT:
  - Implement Python execution first; JIT paths (LLVM/PTX) are optional but should be guarded by availability checks.
  - Avoid introducing unconditional hard dependencies on CUDA or Torch in core execution paths; use markers/feature flags.
- Parameters and numerics:
  - Many parameters are validated and stored as `numpy.ndarray`; adhere to existing patterns (tests enforce array-wrapping for numeric Parameter values).
  - Keep default names stable; registries are cleaned between tests—avoid global mutable state.
- Backward compatibility: avoid breaking public API names and behaviors; discuss major changes before proceeding.

**Tests**
- Co-locate tests under `tests/` mirroring the package area you changed.
- Use pytest markers appropriately (`llvm`, `cuda`, `pytorch`, `stress`).
- Prefer small, deterministic unit tests. Use seeded randomness; conftest seeds numpy/torch and provides helpers.
- When adding components, include tests for:
  - Constructor/parameter validation
  - Execution correctness in Python mode
  - Optional: JIT modes where applicable (guarded with markers)

**Do / Don’t**
- Do:
  - Keep diffs minimal and focused; retain existing style and patterns.
  - Update docs and tests with code changes.
  - Run a targeted subset of tests before pushing; then the full suite if feasible.
  - Respect version pins and CI constraints.
- Don’t:
  - Reformat unrelated files or introduce new tooling (formatters, linters) without prior agreement.
  - Bump dependencies casually or remove pins as part of a feature/bugfix.
  - Introduce global state or nondeterminism into core execution paths.

**CI & Workflows**
- GitHub Actions are configured under `.github/workflows/` to run linting and tests in parallel with strict markers.
- Ensure local changes pass the same checks that run in CI (pytest with configured addopts, including style checks and strict markers).

**Validation Checklist (before opening a PR)**
- Code follows `CONVENTIONS.md` and matches local style.
- New/changed public APIs have docstrings with rST sections; docs build locally.
- Tests added/updated; `pytest -m "not (llvm or cuda or stress)" -n 0` is green; broader suite passes or is justified.
- No unnecessary dependency/version changes.
- New symbols exported via `__init__.py` where appropriate.

**References**
- Coding & docs conventions: `CONVENTIONS.md`
- Contribution workflow: `CONTRIBUTING.md`
- Test configuration and markers: `setup.cfg`, `conftest.py`
- Package metadata and extras: `setup.py`

If anything here conflicts with direct maintainer instructions or issue-specific guidance, follow the maintainer guidance and update this file as needed.

