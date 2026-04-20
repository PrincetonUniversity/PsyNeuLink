# PsyNeuLink TODO

## graph-scheduler FutureWarnings on Python 3.13+

`graph-scheduler` (condition.py lines 582-590) uses `functools.partial` as enum
member values, which Python 3.13+ flags with `FutureWarning`. Currently suppressed
in `notebooks/Getting Started.ipynb` with `warnings.filterwarnings`.

**Action:** When a fixed version of `graph-scheduler` is released, remove the
warning suppression from the starter notebook.

## Switch Makefile to use `uv` for venv and package management

The Makefile currently uses `python3 -m venv` and `pip`. Switching to
[uv](https://github.com/astral-sh/uv) would significantly speed up venv
creation and package installs. This would add `uv` as a prerequisite.

**Action:** Update the Makefile to use `uv venv` and `uv pip install`, and
update GETTING_STARTED.md prerequisites accordingly.
