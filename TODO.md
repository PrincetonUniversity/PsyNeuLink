# PsyNeuLink TODO

## graph-scheduler FutureWarnings on Python 3.13+

`graph-scheduler` (condition.py lines 582-590) uses `functools.partial` as enum
member values, which Python 3.13+ flags with `FutureWarning`. Currently suppressed
in `notebooks/Getting Started.ipynb` with `warnings.filterwarnings`.

**Action:** When a fixed version of `graph-scheduler` is released, remove the
warning suppression from the starter notebook.
